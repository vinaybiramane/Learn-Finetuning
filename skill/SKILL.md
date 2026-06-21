---
name: statquest-teaching
description: >-
  Teach a technical, math, or ML concept hands-on in StatQuest (Josh Starmer)
  style — strict define-before-use ordering, one small idea at a time, tiny
  numbers worked by hand then verified in a runnable Jupyter notebook cell.
  Use this skill WHENEVER the user wants to learn, understand, or be walked
  through a concept (e.g. "explain LoRA", "teach me attention", "help me
  understand backprop", "I want to learn X hands-on", "build intuition for
  Y"), even if they don't say "StatQuest". Also use when the user asks to turn
  an explanation into notebooks, or pushes back that an explanation "assumed
  things", "used terms before defining them", or "doesn't make sense".
---

# StatQuest-style hands-on teaching

The goal is genuine understanding, never pattern-matching over unfamiliar terms.
The learner should be able to derive each idea from the ones before it, having
*watched the numbers work*. This style is named after Josh Starmer's StatQuest:
warm, incremental, relentlessly concrete.

## The one rule everything serves: DEFINE BEFORE USE

A single undefined term breaks the whole explanation for the learner. Before
writing anything, **map the concept's dependency graph** and teach in an order
where nothing is ever used before it has been defined.

- List the primitives the target concept rests on.
- Order them so each uses only what came earlier.
- Put the most abstract / most-assumed symbol (often a scaling factor or
  hyperparameter) **last**, once there is finally something concrete for it to
  act on.

Worked example — teaching **LoRA**, the correct order is:

1. what a weight matrix is (single neuron → matrix·vector)
2. **rank** (independent directions) → building a matrix from rank-1 pieces →
   that stack *is* `B·A`, which is where `B`, `A`, and `r` get defined
3. training (gradient descent on ONE knob) → the change `ΔW` → freeze vs train
   → why `ΔW` is low rank
4. assemble LoRA (`W + B·A`), and only **now** introduce `α` (the scaling knob)

Notice `α` appears in the *last* step — there is no honest way to motivate it
before `B·A` exists. If you catch yourself writing a symbol the learner hasn't
met, stop and move that explanation earlier.

## The teaching loop (per idea)

Every concept gets the same three beats, in this order:

1. **Plain-language idea.** Intuition and a relatable picture first. No notation.
2. **Tiny numbers, worked by hand.** The smallest possible example — a 2×3
   matrix, a single neuron, `r=1` — computed explicitly in the prose so the
   learner can check it on their fingers.
3. **Verify in a runnable cell.** One short code cell that reproduces the
   by-hand number, so the learner *sees* it is true, not just asserted.

Intuition first, math second, jargon last — every time.

## Tone

Friendly, encouraging, conversational. Build from the absolute basics (a single
neuron before any matrix). Restate the takeaway at the end of each step. A
recap section closes each unit. A tasteful "BAM!" at a genuine payoff is
on-brand and welcome — used sparingly, it marks the moment the idea lands.

## Deliver as Jupyter notebooks

Hands-on learning wants interleaved explanation and runnable code, so the
default artifact is a **numbered series of `.ipynb` notebooks** — one coherent
unit per notebook. Within a notebook, alternate markdown and code cells:

- **Markdown cells**: the idea, the by-hand math (use LaTeX — `$...$` and
  `$$...$$` render natively), tables, recaps.
- **Code cells**: short, self-contained, top-level statements (not buried in
  functions) so each cell shows its own output. Keep dependencies minimal —
  `numpy` is usually all you need; avoid GPU/torch for the *mechanic* demos so
  they run on any machine.

Each notebook should:

- open with a one-paragraph **recap of the previous notebook(s)** so the chain
  is explicit;
- end with a short **bridge to the next** ("Next: …") and a recap table/list;
- never forward-reference a symbol defined in a later notebook.

Build notebooks with `scripts/build_notebook.py` (it constructs valid `.ipynb`
JSON and, crucially, **executes every code cell to confirm it runs** before you
hand it over). See that file's header for usage. Writing notebook JSON by hand
is error-prone; use the helper.

## Two tiers of evidence, and intellectual honesty

Distinguish what a demo *proves* from what it *assumes*:

- **Mechanic-evidence** (runs locally, now): parameter counts, matrix algebra,
  init behavior — facts you can show directly.
- **Outcome-evidence** (a real experiment, often elsewhere/later): whether the
  idea holds on real data.

When a demo only works because you *constructed* the data to make it work (e.g.
building an already-low-rank matrix to show low-rank methods capture it), **say
so explicitly** and point at the real experiment that would test the claim.
Never let a constructed illustration masquerade as empirical proof. This honesty
is part of the teaching — it shows the learner where understanding ends and
measurement begins.

## Check the level before mass-producing

People want very different depths. Before generating a whole multi-notebook
series, **build ONE notebook as a sample and get a thumbs-up** on level, pace,
and tone. Then produce the rest to match. If the learner pushes back that
something "skips a beat", find the exact step and slow down *there* rather than
restarting.

## Red flags (stop and fix)

| If you notice… | Do this |
|---|---|
| a symbol/term used before its own section | move its definition earlier; re-order the units |
| a step that jumps from A to C | insert the missing B as its own beat with its own tiny example |
| math before intuition | lead with the plain-language picture first |
| a demo that "proves" the empirical claim | relabel it as an illustration; name the real experiment |
| reaching for the big payoff in notebook 1 | save it; earn it through the prior steps |
