#!/usr/bin/env python3
"""Generate the bundled offline sample knowledge base.

This project ships with an original, offline customer-support dataset for
"Northwind Cloud" (a fictional SaaS product) so the whole pipeline -- ingest,
clean, chunk, embed, retrieve, generate -- can be demoed and tested with zero
network access and zero API keys. It exists *alongside* the real
Hugging Face `HuggingFaceSource` (see
`src/rag_support/data/sources/huggingface_source.py`), which pulls the public
Bitext customer-support dataset when you do have Hub access -- the two
sources implement the same `DataSource` interface, so switching between them
is a one-line config change (`DATA_SOURCE=sample|hf`), not a rewrite.

Run it with `python scripts/generate_sample_dataset.py` to regenerate
`src/rag_support/data/sample_dataset/support_kb_raw.jsonl`. It's checked into
git, so you don't need to run it to use the project -- this script is here
for transparency and so the dataset can be extended.

To make the cleaning stage demonstrably necessary (rather than a no-op),
~15% of records are deliberately corrupted with the kind of mess real
support-ticket exports actually contain: stray HTML, doubled whitespace,
inconsistent casing, and a handful of exact duplicates.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

random.seed(1337)

OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "rag_support"
    / "data"
    / "sample_dataset"
    / "support_kb_raw.jsonl"
)

# Each entry: category, intent, a few paraphrased customer questions, and the
# canonical support answer. Multiple phrasings per intent give the retriever
# (and the eval harness in docs/evaluation.md) more than one way in.
KB: list[dict[str, Any]] = [
    # --- ACCOUNT --------------------------------------------------------
    dict(
        category="ACCOUNT",
        intent="create_account",
        questions=[
            "How do I create a Northwind Cloud account?",
            "I want to sign up for Northwind Cloud, what's the process?",
            "Is there a way to register without a credit card?",
        ],
        answer=(
            "To create a Northwind Cloud account, go to northwindcloud.com/signup and enter "
            "your work email and a password. You'll receive a verification email within a "
            "couple of minutes -- click the link to activate your account. A credit card is "
            "only required if you choose a paid plan; the Free tier needs just an email."
        ),
    ),
    dict(
        category="ACCOUNT",
        intent="reset_password",
        questions=[
            "I forgot my password, how do I reset it?",
            "Can you help me reset my Northwind Cloud password?",
            "The reset password email never arrived, what should I do?",
        ],
        answer=(
            "Click 'Forgot password' on the sign-in page and enter the email on your account. "
            "You'll get a reset link valid for 30 minutes. If it doesn't arrive within a few "
            "minutes, check your spam folder and confirm the email matches your account exactly "
            "-- reset emails are sent from no-reply@northwindcloud.com."
        ),
    ),
    dict(
        category="ACCOUNT",
        intent="update_profile",
        questions=[
            "How can I change the email address on my account?",
            "I need to update my billing name and address.",
        ],
        answer=(
            "Go to Settings > Profile to update your name, billing address, and contact "
            "details. Changing your primary email requires confirming the new address via a "
            "verification link before the change takes effect, to make sure account access "
            "isn't lost."
        ),
    ),
    dict(
        category="ACCOUNT",
        intent="delete_account",
        questions=[
            "How do I permanently delete my Northwind Cloud account?",
            "I want to close my account and remove my data.",
        ],
        answer=(
            "Account deletion is available under Settings > Account > Delete Account. This "
            "immediately cancels any active subscription and schedules your data for permanent "
            "deletion within 30 days, matching our data-retention policy. Deletion cannot be "
            "undone after the 30-day window, so export anything you need first."
        ),
    ),
    dict(
        category="ACCOUNT",
        intent="two_factor_setup",
        questions=[
            "How do I turn on two-factor authentication?",
            "Can I use an authenticator app for 2FA on Northwind Cloud?",
        ],
        answer=(
            "Enable two-factor authentication under Settings > Security > Two-Factor Auth. We "
            "support any TOTP authenticator app (Google Authenticator, Authy, 1Password) via QR "
            "code, plus SMS as a backup method. Save the recovery codes shown during setup "
            "somewhere safe -- they're the only way back in if you lose your device."
        ),
    ),
    # --- BILLING ----------------------------------------------------------
    dict(
        category="BILLING",
        intent="view_invoice",
        questions=[
            "Where can I download my invoices?",
            "I need a PDF receipt for last month's charge.",
        ],
        answer=(
            "All invoices are available under Settings > Billing > Invoice History, where each "
            "one can be downloaded as a PDF. Invoices are also emailed automatically to your "
            "billing contact on the day each charge is processed."
        ),
    ),
    dict(
        category="BILLING",
        intent="update_payment_method",
        questions=[
            "How do I change my credit card on file?",
            "My card expired, how do I update payment details?",
        ],
        answer=(
            "Go to Settings > Billing > Payment Method and click 'Update Card'. The change "
            "applies immediately and will be used for your next billing cycle; any pending "
            "invoice on the old card is retried automatically within 24 hours."
        ),
    ),
    dict(
        category="BILLING",
        intent="dispute_charge",
        questions=[
            "I was charged twice this month, can you help?",
            "There's a charge on my account I don't recognize.",
        ],
        answer=(
            "Sorry about that -- contact billing@northwindcloud.com with the invoice ID (found "
            "under Billing > Invoice History) and we'll investigate within one business day. "
            "Confirmed duplicate or erroneous charges are refunded to the original payment "
            "method within 5-7 business days."
        ),
    ),
    dict(
        category="BILLING",
        intent="cancel_subscription",
        questions=[
            "How do I cancel my subscription?",
            "I want to stop being billed for Northwind Cloud.",
        ],
        answer=(
            "Cancel anytime from Settings > Billing > Manage Plan > Cancel Subscription. You "
            "keep access through the end of the current billing period, and no further charges "
            "are made. Cancelling doesn't delete your data -- use the separate account-deletion "
            "flow for that."
        ),
    ),
    dict(
        category="BILLING",
        intent="apply_coupon",
        questions=[
            "How do I redeem a promo code?",
            "I have a discount coupon, where does it go?",
        ],
        answer=(
            "Enter your coupon code under Settings > Billing > Promotions before your next "
            "renewal date. The discount applies to the following invoice; codes can't be "
            "applied retroactively to invoices already charged."
        ),
    ),
    # --- ORDERS -------------------------------------------------------
    dict(
        category="ORDERS",
        intent="track_order",
        questions=[
            "How do I track my order?",
            "Where is my shipment, it hasn't arrived yet?",
        ],
        answer=(
            "Tracking details are on the Orders page as soon as a shipment leaves our "
            "warehouse, usually within one business day of purchase, and are also emailed to "
            "you automatically. If tracking shows no movement for more than 3 business days, "
            "contact support and we'll follow up with the carrier."
        ),
    ),
    dict(
        category="ORDERS",
        intent="cancel_order",
        questions=[
            "Can I cancel an order I just placed?",
            "I made a mistake on my order, how do I cancel it?",
        ],
        answer=(
            "Orders can be cancelled for free within 60 minutes of purchase from the Orders "
            "page. After that window the order has usually entered fulfillment, so cancellation "
            "isn't possible -- but you can refuse delivery or start a return once it arrives."
        ),
    ),
    dict(
        category="ORDERS",
        intent="modify_order",
        questions=[
            "Can I change the shipping address on an order after placing it?",
            "I need to add an item to an order I already submitted.",
        ],
        answer=(
            "Orders can be modified only during the same 60-minute window as cancellation, from "
            "the Orders page. Once fulfillment has started, the order is locked; you can instead "
            "place a new order for the extra item or redirect the package with the carrier "
            "directly using the tracking link."
        ),
    ),
    dict(
        category="ORDERS",
        intent="delayed_order",
        questions=[
            "My order is late, what happened?",
            "The estimated delivery date has already passed.",
        ],
        answer=(
            "Delays are usually carrier-side; check the tracking link for the latest scan "
            "event. If the estimated delivery date has passed with no update for 3+ business "
            "days, reach out to support with your order number and we'll open a trace with the "
            "carrier or send a replacement, whichever is faster."
        ),
    ),
    # --- SHIPPING ------------------------------------------------------
    dict(
        category="SHIPPING",
        intent="shipping_costs",
        questions=[
            "How much does shipping cost?",
            "Is shipping free on orders over a certain amount?",
        ],
        answer=(
            "Standard shipping is $4.99 and free on orders over $50. Expedited (2-day) "
            "shipping is a flat $12.99 regardless of order size. Exact rates and delivery "
            "windows are shown at checkout before payment."
        ),
    ),
    dict(
        category="SHIPPING",
        intent="change_shipping_address",
        questions=[
            "I entered the wrong shipping address, can it be fixed?",
            "How do I update my delivery address for an order in progress?",
        ],
        answer=(
            "If the order was placed within the last 60 minutes, update the address directly "
            "from the Orders page. After that, contact support immediately -- we can sometimes "
            "redirect a package with the carrier before the first delivery attempt, but this "
            "isn't guaranteed once it's out for delivery."
        ),
    ),
    dict(
        category="SHIPPING",
        intent="international_shipping",
        questions=[
            "Do you ship internationally?",
            "What countries can Northwind Cloud hardware be shipped to?",
        ],
        answer=(
            "We currently ship to the US, Canada, the UK, and the EU. International orders may "
            "be subject to customs duties and import taxes charged by your local authority, "
            "which are not included in the checkout price."
        ),
    ),
    # --- RETURNS ---------------------------------------------------------
    dict(
        category="RETURNS",
        intent="return_policy",
        questions=[
            "What is your return policy?",
            "How many days do I have to return something?",
        ],
        answer=(
            "Items can be returned within 30 days of delivery in their original condition and "
            "packaging for a full refund. Digital goods and gift cards are not eligible for "
            "returns. Return shipping is free for defective items and $5.99 otherwise, deducted "
            "from the refund."
        ),
    ),
    dict(
        category="RETURNS",
        intent="start_return",
        questions=[
            "How do I start a return?",
            "I want to send an item back, what's the process?",
        ],
        answer=(
            "Go to Orders, select the item, and click 'Start Return' to generate a prepaid "
            "shipping label. Drop the package at any partner carrier location within 14 days of "
            "generating the label; refunds are issued within 5 business days of the return "
            "being received at our warehouse."
        ),
    ),
    dict(
        category="RETURNS",
        intent="refund_status",
        questions=[
            "Where is my refund?",
            "It's been a week since my return arrived, when do I get refunded?",
        ],
        answer=(
            "Refunds are processed within 5 business days of the returned item arriving at our "
            "warehouse, and can take an additional 3-5 business days to appear on your statement "
            "depending on your bank. You'll get an email confirmation the moment the refund is "
            "issued on our end."
        ),
    ),
    dict(
        category="RETURNS",
        intent="exchange_item",
        questions=[
            "Can I exchange an item for a different size instead of a refund?",
            "How do I swap a defective unit for a working one?",
        ],
        answer=(
            "Exchanges are handled as a return plus a new order: start a return for the "
            "original item, and we'll ship the replacement as soon as the return label is "
            "scanned by the carrier, without waiting for the item to physically arrive back at "
            "our warehouse."
        ),
    ),
    # --- TECHNICAL ----------------------------------------------------
    dict(
        category="TECHNICAL",
        intent="login_issue",
        questions=[
            "I can't log in even with the right password.",
            "Getting an 'invalid credentials' error but I'm sure the password is correct.",
        ],
        answer=(
            "This is usually caused by a stale session or an autofilled password with a "
            "trailing space. Try clearing the password field and retyping it, or use an "
            "incognito window. If two-factor authentication was recently enabled, make sure "
            "you're entering the current 6-digit code, not a saved recovery code."
        ),
    ),
    dict(
        category="TECHNICAL",
        intent="app_crash",
        questions=[
            "The app keeps crashing when I open the dashboard.",
            "Northwind Cloud desktop app closes immediately on launch.",
        ],
        answer=(
            "First, update to the latest version from Settings > About -- most launch crashes "
            "are fixed in the newest release. If it still crashes, delete the local cache folder "
            "(Help > Open Cache Folder) and relaunch; this rebuilds the local index without "
            "touching your account data."
        ),
    ),
    dict(
        category="TECHNICAL",
        intent="sync_issue",
        questions=[
            "My changes aren't syncing across devices.",
            "Data on my phone doesn't match what's on the web app.",
        ],
        answer=(
            "Sync runs automatically every 30 seconds when a device is online. Check Settings > "
            "Sync Status on each device for the last successful sync time; if one device shows "
            "an error, sign out and back in to force a full resync. Sync is paused entirely "
            "while offline and resumes automatically on reconnect."
        ),
    ),
    dict(
        category="TECHNICAL",
        intent="api_error",
        questions=[
            "I'm getting a 429 error from the Northwind Cloud API.",
            "Why am I seeing rate limit errors on the REST API?",
        ],
        answer=(
            "A 429 means you've hit the rate limit: 120 requests per minute on the Free tier, "
            "1,200 on paid plans. Use the `Retry-After` response header to back off "
            "automatically, and consider batching requests via the `/v1/batch` endpoint if you "
            "regularly approach the limit."
        ),
    ),
    dict(
        category="TECHNICAL",
        intent="feature_request",
        questions=[
            "How do I suggest a new feature?",
            "Is there a public roadmap I can add ideas to?",
        ],
        answer=(
            "Feature requests go through feedback.northwindcloud.com, where you can submit an "
            "idea or upvote existing ones -- our product team reviews the top-voted items every "
            "sprint. There's no guaranteed response time, but shipped requests are tagged "
            "'Delivered' with a link to the release notes."
        ),
    ),
    # --- SUBSCRIPTION --------------------------------------------------
    dict(
        category="SUBSCRIPTION",
        intent="upgrade_plan",
        questions=[
            "How do I upgrade from the Free plan to Pro?",
            "What's the difference between the Pro and Team plans?",
        ],
        answer=(
            "Upgrade anytime from Settings > Billing > Manage Plan; the new plan's features "
            "activate immediately and you're billed a prorated amount for the rest of the "
            "current cycle. Pro adds unlimited projects and priority support; Team adds shared "
            "workspaces, roles/permissions, and SSO."
        ),
    ),
    dict(
        category="SUBSCRIPTION",
        intent="downgrade_plan",
        questions=[
            "How do I downgrade my plan?",
            "I want to move from Team back to Pro.",
        ],
        answer=(
            "Downgrades take effect at the start of your next billing cycle rather than "
            "immediately, so you keep current-plan features until then. If you're over the "
            "lower plan's limits (e.g. too many active projects), you'll be prompted to archive "
            "some before the downgrade completes."
        ),
    ),
    dict(
        category="SUBSCRIPTION",
        intent="pause_subscription",
        questions=[
            "Can I pause my subscription instead of cancelling?",
            "I won't need the account for a couple of months, is there a hold option?",
        ],
        answer=(
            "Yes -- Settings > Billing > Manage Plan > Pause Subscription suspends billing for "
            "up to 3 months while keeping your data and configuration intact. Your workspace "
            "becomes read-only while paused and reactivates automatically at the end of the "
            "pause period."
        ),
    ),
    dict(
        category="SUBSCRIPTION",
        intent="renewal_info",
        questions=[
            "When does my subscription renew?",
            "Will I get a notice before I'm charged for renewal?",
        ],
        answer=(
            "Your renewal date is shown under Settings > Billing > Manage Plan, and we send a "
            "reminder email 7 days before every renewal charge. Annual plans also get a second "
            "reminder 30 days out, since the charge is larger."
        ),
    ),
    # --- PRIVACY --------------------------------------------------------
    dict(
        category="PRIVACY",
        intent="data_export",
        questions=[
            "How do I export all of my data?",
            "Can I get a copy of everything stored on my account?",
        ],
        answer=(
            "Request a full export from Settings > Privacy > Export My Data. We compile a "
            "downloadable archive (JSON + CSV) within 48 hours and email you a secure link that "
            "stays active for 7 days."
        ),
    ),
    dict(
        category="PRIVACY",
        intent="data_deletion",
        questions=[
            "How do I request deletion of my personal data?",
            "I want my data wiped, not just my account closed.",
        ],
        answer=(
            "Data deletion requests can be submitted from Settings > Privacy > Delete My Data, "
            "independent of closing your account. We complete deletion within 30 days as "
            "required by our data-retention policy, except for records we're legally required "
            "to keep (e.g. billing records for tax purposes)."
        ),
    ),
    dict(
        category="PRIVACY",
        intent="gdpr_request",
        questions=[
            "How do I submit a GDPR data subject access request?",
            "I'm an EU resident, what are my data rights here?",
        ],
        answer=(
            "EU/UK users can submit access, correction, or erasure requests to "
            "privacy@northwindcloud.com or via Settings > Privacy. We respond within the "
            "GDPR-mandated 30-day window, and our Data Processing Addendum is available on "
            "request for business customers."
        ),
    ),
    # --- CONTACT ----------------------------------------------------------
    dict(
        category="CONTACT",
        intent="contact_support",
        questions=[
            "How do I reach a human support agent?",
            "Is there a phone number for Northwind Cloud support?",
        ],
        answer=(
            "Live chat is available from the in-app Help widget, staffed 24/5 (Mon-Fri). Email "
            "support@northwindcloud.com for anything non-urgent -- typical first response time "
            "is under 4 hours on business days. Phone support is available on Team and "
            "Enterprise plans only."
        ),
    ),
    dict(
        category="CONTACT",
        intent="business_hours",
        questions=[
            "What are your support hours?",
            "Is support available on weekends?",
        ],
        answer=(
            "Live chat and phone support run Monday-Friday, 6am-6pm Pacific Time. Email support "
            "is monitored on weekends with a longer response time (typically by end of the next "
            "business day)."
        ),
    ),
    dict(
        category="CONTACT",
        intent="escalate_issue",
        questions=[
            "My issue hasn't been resolved, how do I escalate it?",
            "Can I speak to a manager about my support ticket?",
        ],
        answer=(
            "Reply to your existing ticket with 'ESCALATE' in the subject line, or ask your "
            "support agent directly -- either routes the ticket to a team lead within one "
            "business hour. Enterprise customers can also escalate through their named account "
            "manager for faster turnaround."
        ),
    ),
]


def _messify(text: str, rng: random.Random) -> str:
    """Corrupt a clean string the way a real ticket-export pipeline would."""
    choices = [
        lambda t: f"<p>{t}</p>",
        lambda t: t.replace(" ", "  ", 2),
        lambda t: t.upper() if rng.random() < 0.3 else t.lower(),
        lambda t: f"  {t}  \n\n",
        # Escape & first, then quotes -- doing it in the other order would
        # re-escape the "&" just introduced by the quote replacement and
        # produce invalid double-escaped entities like "&amp;#39;".
        lambda t: t.replace("&", "&amp;").replace("'", "&#39;"),
    ]
    n = rng.randint(1, 2)
    for fn in rng.sample(choices, n):
        text = fn(text)
    return text


def build_records() -> list[dict[str, Any]]:
    rng = random.Random(1337)
    records: list[dict[str, Any]] = []
    counter = 0
    for entry in KB:
        for q in entry["questions"]:
            counter += 1
            text = f"Customer: {q}\nSupport: {entry['answer']}"
            noisy = rng.random() < 0.15
            record = {
                "external_id": f"kb-{counter:04d}",
                "category": entry["category"],
                "intent": entry["intent"],
                "title": q,
                "text": _messify(text, rng) if noisy else text,
                "metadata": {"is_synthetic_noise": noisy},
            }
            records.append(record)

    # Sprinkle in a handful of exact duplicates -- common in real ticket
    # exports when the same canned answer is logged more than once -- so the
    # cleaning stage's de-duplication step has something real to remove.
    for dupe_source in rng.sample(records, 6):
        records.append(dict(dupe_source, external_id=dupe_source["external_id"] + "-dup"))

    rng.shuffle(records)
    return records


def main() -> None:
    records = build_records()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
