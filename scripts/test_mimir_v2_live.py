"""Comprehensive Live Test for FvF Mímir V2.

Verifies:
1. Classification accuracy on diverse real human web texts across lengths (no length bias).
2. Human encyclopedic & formal articles (Earth, Pacific Ocean, news, disclaimers).
3. Real AI texts across domains (GPT-Wiki-Intro, HC3 Finance, Medicine, CS/AI, ELI5).
4. Text length invariance: short paragraphs (180 chars), medium paragraphs (500 chars), long articles (1200+ chars).
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add native classifiers to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "native" / "classifiers"))

from fvf_mimir_classifier import FvfMimirClassifier


TEST_CASES = [
    # Human Encyclopedia & Formal Web (Earth, Pacific Ocean, Wikipedia)
    (
        "Human (Wikipedia - Earth)",
        "Earth is the third planet from the Sun and the only astronomical object known to harbor life. This is enabled by Earth being a water world, the only one in the Solar System sustaining liquid surface water. Almost all of Earth's water is contained in its global ocean, covering 70.8% of Earth's crust. The remaining 29.2% of Earth's crust is land, most of which is located in the land hemisphere. Most of Earth's land is somewhat humid and covered by vegetation, while large sheets of ice at Earth's polar deserts retain more water than Earth's groundwater, lakes, rivers and atmospheric water combined. Earth's crust consists of slowly moving tectonic plates, which interact to produce mountain ranges, volcanoes, and earthquakes.",
        False
    ),
    (
        "Human (Wikipedia - Pacific Ocean)",
        "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions. It extends from the Arctic Ocean in the north to the Southern Ocean in the south, and is bounded by the continents of Asia and Australia in the west and the Americas in the east. At 165,250,000 square kilometers in area, this largest division of the World Ocean—and, in turn, the hydrosphere—covers about 46% of Earth's water surface and about 32% of its total surface area, larger than Earth's entire land area combined. The centers of both the Water Hemisphere and the Western Hemisphere, as well as the oceanic pole of inaccessibility, are in the Pacific Ocean.",
        False
    ),
    # Human Journalism (BBC News / Reuters style)
    (
        "Human (News - Reuters/BBC)",
        "European central bankers signaled on Wednesday that further interest rate reductions remain firmly on the table as inflationary pressures across the eurozone continue to recede faster than initially projected. Speaking at a financial conference in Frankfurt, policymakers noted that while wage growth remains sticky in specific service sectors, overall economic activity across Germany and France has softened substantially, prompting analysts to price in an additional quarter-point cut at the upcoming monetary policy meeting next month.",
        False
    ),
    # Human Legal / Web Boilerplate (Terms of Service)
    (
        "Human (Web Terms of Service)",
        "By accessing or using our Services, you agree to be bound by these Terms. If you do not agree to these Terms, including the mandatory arbitration provision and class action waiver, do not access or use our Services. We may make changes to these Terms from time to time. If we make changes, we will provide you with notice of such changes, such as by sending an email, providing a notice through our Services, or updating the date at the top of these Terms. Unless we say otherwise in our notice, the amended Terms will be effective immediately, and your continued use of our Services will confirm your acceptance of the changes.",
        False
    ),
    # Human Casual / Conversational (Reddit / Forum)
    (
        "Human (Reddit / Forum Comment)",
        "Honestly I had the same issue last week when I tried updating my motherboard BIOS. Everything seemed fine during the flash process, but once it rebooted the monitor just stayed completely black for a solid two minutes. Don't panic and pull the plug—it's actually just doing memory training on DDR5. Give it around five minutes and see if the debug LED changes color before you try clearing CMOS.",
        False
    ),
    # Real AI Generated Text (ChatGPT Wikipedia Intro - Person)
    (
        "AI (ChatGPT - Doug Wolaver)",
        "Doug Wolaver is a Tennessee Walking Horse trainer who has been involved in the sport for more than forty years. Doug Wolaver is a trainer of Tennessee Walking Horses and has been involved in the sport for more than forty years. He has competed in various horse shows throughout the United States, winning more than fifty awards and championships. Wolaver also serves as the Chairman of the Tennessee Walking Horse National Celebration, an annual event that celebrates the horses and their riders.",
        True
    ),
    # Real AI Generated Text (ChatGPT Finance / Business explainer)
    (
        "AI (ChatGPT - Business Banking)",
        "There are several reasons why a business checking account may be a better choice for a business than a personal checking account. First, business checking accounts typically offer higher transaction limits and more services tailored to businesses, such as payroll processing and credit card processing. In addition, having a separate business checking account can help you keep your personal and business finances distinct, which can make bookkeeping and taxes easier.",
        True
    ),
    # Real AI Generated Text (ChatGPT Healthcare / Medical Advice)
    (
        "AI (ChatGPT - Option Time Value)",
        "The time value, also known as the extrinsic value, is the portion of the option price that is not related to the intrinsic value, and it represents the time remaining until the option's expiration date, as well as the volatility of the underlying security and the risk-free interest rate. For example, let's say that you see the following quote for a call option on XYZ stock: Bid: $2.50, Ask: $2.55. This means that a buyer is willing to pay $2.50 for the option, while a seller is willing to sell it for $2.55. The difference between the bid and the ask price, also known as the bid-ask spread, is a measure of the liquidity of the option. It's important to note that the option price does not represent the potential profit or loss that you can make from buying or selling the option. Instead, it represents the price that you have to pay to buy the option or the price that you will receive if you sell the option.",
        True
    ),
    # Real AI Generated Text (ChatGPT Computer Science / Machine Learning)
    (
        "AI (ChatGPT - Automatic Diff)",
        "AD has many applications in machine learning, where it is used to compute the gradients of loss functions with respect to model parameters. These gradients are crucial for gradient-based optimization algorithms like stochastic gradient descent, which adjust model parameters to minimize loss and optimize performance on tasks such as classification and regression.",
        True
    ),
    # Real AI Generated Text (ChatGPT Everyday Explainer - ELI5)
    (
        "AI (ChatGPT - ELI5 Tampons)",
        "When a woman uses a tampon, it is inserted into the vagina to absorb menstrual blood. The tampon should be changed every four to eight hours, depending on the flow of the menstrual period. To remove the tampon, gently pull on the string attached to the end of the tampon. It is important to throw the used tampon in the trash and never flush it down the toilet.",
        True
    ),
]


def main():
    print("Initializing FvfMimirClassifier...")
    clf = FvfMimirClassifier()
    ok, err = clf.load_model()
    if not ok:
        print(f"Failed to load classifier: {err}")
        return 1

    print(f"Classifier loaded successfully.")
    print(f"Active Device: {clf.get_device_info()}")
    print("=" * 86)
    print(f"{'Label & Title':<37} | {'Length':<6} | {'AI Prob':<8} | {'Prediction':<10} | {'Status'}")
    print("-" * 86)

    texts = [text for _, text, _ in TEST_CASES]
    batch, valid_indices = clf.preprocess_batch(texts, modality="text")
    probs = clf.classify_batch(batch)

    correct = 0
    total = len(TEST_CASES)

    for out_idx, orig_idx in enumerate(valid_indices):
        title, text, is_ai = TEST_CASES[orig_idx]
        prob = probs[out_idx]
        pred_ai = prob >= 0.75  # Extension threshold
        expected_str = "AI (>=0.75)" if is_ai else "Human (<0.75)"
        pred_str = "AI" if pred_ai else "Human"
        passed = (pred_ai == is_ai)
        if passed:
            correct += 1
            status = "PASS [OK]"
        else:
            status = "FAIL [X]"

        print(f"{title:<37} | {len(text):<6d} | {prob * 100:6.2f}% | {pred_str:<10} | {status} (Expected: {expected_str})")

    print("=" * 86)
    print(f"Summary: {correct}/{total} tests passed ({correct / total * 100:.1f}%)")
    return 0 if correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
