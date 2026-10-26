"""FraudShield: agent-style CLI for credit card fraud detection.

Usage:
    uv run cli.py            # check a transaction (same as `predict`)
    uv run cli.py predict    # check a transaction
    uv run cli.py metrics    # show the saved model's scores
    uv run cli.py demo       # run two sample transactions
"""

from __future__ import annotations

import calendar
import time
from datetime import date, datetime

import joblib
import questionary
import typer
import xgboost as xgb
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from train import (
    FEATURE_LABELS,
    MODEL_PATH,
    REPORTS_DIR,
    comparison_table,
    confusion_table,
    cv_table,
    fmt_score,
    test_table,
    transaction_features,
)

console = Console()
app = typer.Typer(add_completion=False, help="FraudShield: credit card fraud detection agent.")

STEP_DELAY = 0.35  # small pause so each agent step is readable

CATEGORY_LABELS = {
    "entertainment": "Entertainment",
    "food_dining": "Food & Dining",
    "gas_transport": "Fuel & Transport",
    "grocery_net": "Grocery (online)",
    "grocery_pos": "Grocery (in store)",
    "health_fitness": "Health & Fitness",
    "home": "Home",
    "kids_pets": "Kids & Pets",
    "misc_net": "Miscellaneous (online)",
    "misc_pos": "Miscellaneous (in store)",
    "personal_care": "Personal Care",
    "shopping_net": "Shopping (online)",
    "shopping_pos": "Shopping (in store)",
    "travel": "Travel",
}

RECOMMENDATIONS = {
    "HIGH": "Block this transaction and contact the cardholder on their registered mobile number.",
    "MEDIUM": "Hold the transaction and ask for extra verification (OTP or a call) before approving.",
    "LOW": "This looks like a normal transaction. Safe to approve.",
}

PROMPT_STYLE = questionary.Style([
    ("qmark", "fg:#00d7ff bold"),
    ("question", "bold"),
    ("answer", "fg:#00d7ff bold"),
    ("pointer", "fg:#00d7ff bold"),
    ("highlighted", "fg:#00d7ff bold"),
    ("instruction", "fg:#808080"),
])


# -------------------------------------------------------------------- card --

def luhn_valid(number: str) -> bool:
    total = 0
    for i, digit in enumerate(reversed(number)):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def card_network(number: str) -> str | None:
    """Detect the card network from its prefix (networks used in India)."""
    if number.startswith(("34", "37")):
        return "American Express"
    if number.startswith(("508", "60", "65", "81", "82", "353", "356")):
        return "RuPay"
    if number.startswith("36") or 300 <= int(number[:3]) <= 305:
        return "Diners Club"
    if 51 <= int(number[:2]) <= 55 or 2221 <= int(number[:4]) <= 2720:
        return "Mastercard"
    if number.startswith("4"):
        return "Visa"
    return None


def clean_card(text: str) -> str:
    return text.replace(" ", "").replace("-", "")


def mask_card(number: str) -> str:
    return f"•••• •••• •••• {number[-4:]}"


def validate_card(text: str) -> bool | str:
    number = clean_card(text)
    if not number.isdigit() or not 13 <= len(number) <= 19:
        return "Enter 13 to 19 digits"
    if card_network(number) is None:
        return "Card network not supported (RuPay, Visa, Mastercard, Amex, Diners)"
    if not luhn_valid(number):
        return "Invalid card number (Luhn check failed)"
    return True


# ------------------------------------------------------------------- input --

def validate_number(minimum: float = 0, integer: bool = False):
    def check(text: str) -> bool | str:
        try:
            value = int(text) if integer else float(text.replace(",", ""))
        except ValueError:
            return "Enter a whole number" if integer else "Enter a number"
        return True if value >= minimum else f"Must be at least {minimum}"
    return check


def validate_datetime(text: str) -> bool | str:
    try:
        datetime.strptime(text, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Use format YYYY-MM-DD HH:MM"
    return True


def validate_dob(text: str) -> bool | str:
    try:
        dob = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return "Use format YYYY-MM-DD"
    age = (date.today() - dob).days / 365.25
    return True if 18 <= age <= 100 else "Cardholder age must be between 18 and 100"


def ask_text(message: str, validate, default: str = "") -> str:
    return questionary.text(message, default=default, validate=validate, qmark="◆", style=PROMPT_STYLE).unsafe_ask()


def ask_select(message: str, choices, default=None):
    return questionary.select(message, choices=choices, default=default, qmark="◆", style=PROMPT_STYLE).unsafe_ask()


def collect_transaction(categories: list[str]) -> dict:
    say("Tell me about the transaction you want to check.")
    console.print()

    card = clean_card(ask_text("Card number", validate_card))
    amount = float(ask_text("Amount (₹)", validate_number(1)).replace(",", ""))
    category = ask_select(
        "Merchant category",
        [questionary.Choice(CATEGORY_LABELS.get(c, c), value=c) for c in categories],
    )

    if questionary.confirm("Did it happen just now?", default=True, qmark="◆", style=PROMPT_STYLE).unsafe_ask():
        when = datetime.now().replace(second=0, microsecond=0)
    else:
        when = datetime.strptime(ask_text("Date and time (YYYY-MM-DD HH:MM)", validate_datetime), "%Y-%m-%d %H:%M")

    dob = datetime.strptime(ask_text("Cardholder date of birth (YYYY-MM-DD)", validate_dob), "%Y-%m-%d").date()

    console.print()
    say("A few questions about recent card activity.")
    console.print()
    avg = float(ask_text("Usual amount per transaction on this card (₹)", validate_number(1), "2500").replace(",", ""))
    count = int(ask_text("Transactions in the last 24 hours (not counting this one)", validate_number(0, True), "1"))
    gap = float(ask_text("Hours since the previous transaction", validate_number(0), "12"))

    return {
        "card": card, "amount": amount, "category": category, "when": when, "dob": dob,
        "avg": avg, "count": count, "gap": gap,
    }


# ------------------------------------------------------------------- agent --

def say(message: str) -> None:
    console.print(f"[bold cyan]●[/] {message}")


def stream(message: str, style: str = "") -> None:
    """Print text a few characters at a time, like an AI agent typing."""
    console.print("[bold cyan]●[/] ", end="")
    for ch in message:
        console.print(ch, end="", style=style, markup=False, highlight=False)
        time.sleep(0.012)
    console.print()


def run_step(title: str, fn):
    """Run one agent step with a spinner, then print a done line."""
    with console.status(f"[cyan]{title}...", spinner="dots"):
        result, detail = fn()
        time.sleep(STEP_DELAY)
    console.print(f"  [green]✓[/] {title}  [dim]{detail}[/]")
    return result


def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        console.print(Panel(
            "Model not found. Train it first:\n\n  [bold]uv run train.py[/]",
            title="FraudShield", border_style="red",
        ))
        raise typer.Exit(1)
    with console.status("[cyan]Loading model...", spinner="dots"):
        return joblib.load(MODEL_PATH)


def banner(bundle: dict) -> None:
    m = bundle["metrics"]["test"]["tuned"]
    console.print(Panel(
        Group(
            Text.from_markup("[bold cyan]◆ FraudShield[/]  [dim]credit card fraud detection agent[/]"),
            Text.from_markup(
                f"[dim]{bundle['model_name']} · v{bundle['version']} · trained {bundle['trained_on'][:10]} · "
                f"recall {m['recall']:.2%} · precision {m['precision']:.2%}[/]"
            ),
        ),
        border_style="cyan", padding=(0, 2),
    ))


def describe(feature: str, value) -> str:
    match feature:
        case "amount_inr":
            return f"₹{value:,.0f}"
        case "hour":
            return f"{int(value):02d}:00"
        case "is_night":
            return "Yes" if value else "No"
        case "day_of_week":
            return calendar.day_name[int(value)]
        case "age":
            return f"{value:.0f} years"
        case "txn_count_24h":
            return f"{int(value)}"
        case "hours_since_last":
            return f"{value:.1f} h"
        case "amt_to_avg_ratio":
            return f"{value:.1f}x usual"
        case "category":
            return CATEGORY_LABELS.get(value, value)
    return str(value)


def explain(pipeline, row, top: int = 4) -> list[tuple[str, float]]:
    """Return (feature, contribution) pairs that pushed the prediction most.

    For XGBoost this uses per-feature contributions (SHAP values). One-hot
    category columns are added back into a single `category` feature.
    """
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    X = prep.transform(row)
    if hasattr(model, "get_booster"):
        values = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0][:-1]
    elif hasattr(model, "coef_"):
        values = model.coef_[0] * X[0]
    else:
        values = model.feature_importances_ * 0  # no direction for plain trees
    totals: dict[str, float] = {}
    for name, value in zip(prep.get_feature_names_out(), values):
        base = "category" if name.startswith("cat__") else name.removeprefix("num__")
        totals[base] = totals.get(base, 0.0) + float(value)
    return sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top]


def risk_level(proba: float, threshold: float) -> tuple[str, str]:
    if proba >= threshold:
        return "HIGH", "red"
    if proba >= threshold * 0.5:
        return "MEDIUM", "yellow"
    return "LOW", "green"


def risk_meter(proba: float, color: str, width: int = 32) -> str:
    filled = round(proba * width)
    return f"[{color}]{'█' * filled}[/][dim]{'░' * (width - filled)}[/] [bold {color}]{proba:.1%}[/]"


def analyze(bundle: dict, tx: dict) -> None:
    pipeline, threshold = bundle["pipeline"], bundle["threshold"]
    console.print()
    say("Analyzing the transaction.")

    network = run_step(
        "Validating card",
        lambda: (card_network(tx["card"]), f"{card_network(tx['card'])} {mask_card(tx['card'])} · Luhn check passed"),
    )

    def build():
        row = transaction_features(
            amount_inr=tx["amount"], category=tx["category"], when=tx["when"], dob=tx["dob"],
            txn_count_24h=tx["count"], hours_since_last=tx["gap"], avg_amount_inr=tx["avg"],
        )
        r = row.iloc[0]
        return row, (f"age {r.age:.0f} · {r.amt_to_avg_ratio:.1f}x usual spend · "
                     f"{r.txn_count_24h} txns in 24h · night {'yes' if r.is_night else 'no'}")

    row = run_step("Building features", build)
    proba = run_step(
        f"Running {bundle['model_name']} model",
        lambda: (p := float(pipeline.predict_proba(row)[0, 1]), f"fraud probability {p:.4f}"),
    )
    reasons = run_step("Explaining the decision", lambda: (r := explain(pipeline, row), f"top {len(r)} factors"))

    level, color = risk_level(proba, threshold)
    verdict = {"HIGH": "Likely FRAUD", "MEDIUM": "Needs a closer look", "LOW": "Looks genuine"}[level]
    icon = {"HIGH": "✖", "MEDIUM": "▲", "LOW": "✔"}[level]

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("Card", f"{network} {mask_card(tx['card'])}")
    summary.add_row("Amount", f"₹{tx['amount']:,.2f} · {CATEGORY_LABELS.get(tx['category'], tx['category'])}")
    summary.add_row("When", tx["when"].strftime("%d %b %Y, %I:%M %p"))
    summary.add_row("Activity", f"{tx['count']} txns in last 24h · last one {tx['gap']:g} h ago")
    summary.add_row("Risk", risk_meter(proba, color))
    summary.add_row("Threshold", f"{threshold:.3f}")

    factors = Table(box=None, padding=(0, 2), show_edge=False)
    factors.add_column("Factor", style="bold")
    factors.add_column("Value")
    factors.add_column("Effect")
    for feature, value in reasons:
        effect = "[red]▲ raises risk[/]" if value > 0 else "[green]▼ lowers risk[/]" if value < 0 else "[dim]key factor[/]"
        factors.add_row(FEATURE_LABELS.get(feature, feature), describe(feature, row.iloc[0][feature]), effect)

    console.print()
    console.print(Panel(
        Group(summary, Text(""), Text("Why", style="bold underline"), factors),
        title=f"[bold {color}] {icon} {level} RISK · {verdict} [/]",
        title_align="left", border_style=color, padding=(1, 2),
    ))
    stream(RECOMMENDATIONS[level], style=f"bold {color}")


# ---------------------------------------------------------------- commands --

@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        predict()


@app.command()
def predict() -> None:
    """Check a transaction step by step."""
    bundle = load_bundle()
    banner(bundle)
    try:
        while True:
            analyze(bundle, collect_transaction(bundle["categories"]))
            console.print()
            if not questionary.confirm("Check another transaction?", default=True, qmark="◆", style=PROMPT_STYLE).unsafe_ask():
                break
            console.print()
    except KeyboardInterrupt:
        console.print()
    say("Goodbye. Stay safe.")


@app.command()
def metrics() -> None:
    """Show the saved model's accuracy measures."""
    bundle = load_bundle()
    banner(bundle)
    report = bundle["metrics"]
    data = report["dataset"]

    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column()
    info.add_row("Model", f"{report['model_name']} v{report['version']}")
    info.add_row("Trained on", report["trained_on"])
    info.add_row("Dataset", data["source"])
    info.add_row("Rows", f"train {data['train_rows']:,} · test {data['test_rows']:,}")
    info.add_row("Fraud rate", f"{data['fraud_rate']:.2%}")
    info.add_row("Threshold", f"{report['threshold']:.3f} (tuned for best F1)")
    info.add_row("Test PR-AUC", fmt_score(report["test"]["tuned"]["pr_auc"]))
    console.print(Panel(info, title="[bold]Model info[/]", title_align="left", border_style="cyan"))

    console.print(test_table(report))
    console.print(cv_table(report["cv"]))
    console.print(confusion_table(report["test"]["tuned"]["confusion_matrix"]))
    console.print(comparison_table(report["comparison"], report["model_name"]))

    graphs = sorted(REPORTS_DIR.glob("*.png"))
    if graphs:
        console.print(Panel(
            "\n".join(f"[cyan]•[/] {p}" for p in graphs),
            title="[bold]Graphs[/]", title_align="left", border_style="dim",
        ))


@app.command()
def demo() -> None:
    """Run one normal and one suspicious sample transaction."""
    bundle = load_bundle()
    banner(bundle)
    samples = [
        ("a normal grocery purchase", {
            "card": "6521501234567893", "amount": 1450.0, "category": "grocery_pos",
            "when": datetime(2026, 10, 24, 18, 20), "dob": date(1990, 5, 14),
            "avg": 2000.0, "count": 1, "gap": 20.0,
        }),
        ("a suspicious late night online purchase", {
            "card": "5123456789012346", "amount": 82000.0, "category": "shopping_net",
            "when": datetime(2026, 10, 25, 23, 10), "dob": date(1971, 2, 3),
            "avg": 7000.0, "count": 5, "gap": 0.5,
        }),
    ]
    for i, (label, tx) in enumerate(samples, start=1):
        console.print()
        console.rule(f"[bold]Demo {i}/{len(samples)}[/]", style="dim")
        say(f"Checking {label}.")
        analyze(bundle, tx)


if __name__ == "__main__":
    app()
