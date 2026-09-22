# Beyond Bought-Together

Actionable cross-sell versus popularity in grocery baskets. A semester project for CSCI 5502 Data Mining at the University of Colorado Boulder.

## Team

- Bhaveeasheshwar Eswaran (Bhaveeash)
- Raghavendra Kumaran

## Project goal

Grocery recommendations ranked by raw co-occurrence resurface staples and blur the line between scheduled replenishment and genuine cross-sell, which wastes scarce recommendation slots and nudges. This project separates actionable cross-sell from popularity and from replenishment, and routes each association to the intervention it warrants, whether that is a bundle, a timed reminder, a next-trip nudge, or nothing at all.

## Research questions

1. How much of the standard "frequently bought together" ranking is really about popularity rather than genuine association?
2. Which associations are actionable cross-sell, meaning strong under a null-invariant measure where the paired item is not a near-universal staple, as opposed to popularity-driven co-occurrence? This is the primary question.
3. Among recurring purchases, can the regularity of the gaps between buys separate scheduled replenishment from complementary co-purchase, and how much of purchase volume falls in each?
4. Do the items a customer buys on one visit predict different items on later visits, revealing cross-sell moments that same-basket analysis misses?
5. Which associations flagged as actionable actually hold up across time, so we can separate durable signal from sampling noise?

## Data

Groceries Dataset by Heeral Dedhia on Kaggle, at https://www.kaggle.com/datasets/heeraldedhia/groceries-dataset. About 38,765 rows, roughly 3,900 members, and around 167 items, spanning 2014 and 2015. It has three fields, which are Member_number, Date, and itemDescription. A basket is defined as all items that share a member and a date, since the dataset has no checkout ID. The CSV is free and public and small enough to run on a laptop.

The raw CSV is not committed to this repository. Download instructions live in `data/`, along with a small non-sensitive sample for a quick look.

## Repository structure

```
data/          acquisition instructions and a small non-sensitive sample (raw CSV not committed)
notebooks/     exploration and staged analysis
src/           reusable code for basket construction, measures, and sequential mining
figures/       generated charts from the analysis
report/        milestone write-ups
docs/          the project website (GitHub Pages)
docs/figures/  images used by the website, including the routing diagram
README.md
.gitignore
requirements.txt
```

## Website

The project website is published with GitHub Pages from the `docs/` folder. Live link to be added once Pages is enabled.

## Milestone roadmap

- Milestone 0: project idea and framing, approved.
- Milestone 1: framing, research questions, team, website, and repository setup. This is the current milestone.
- Later milestones: data understanding and cleaning, association strength and the popularity split, replenishment and sequential analysis, temporal validation, and the final routed rule set.

## Reproducibility

Pin the environment with `requirements.txt` so the analysis re-runs cleanly. Do not commit credentials, private data, or anything that could re-identify a member. Results are reported only at the aggregate rule level.

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies with `pip install -r requirements.txt`.
4. Follow the instructions in `data/` to download the dataset.
