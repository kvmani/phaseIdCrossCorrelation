# Presentation Authoring Guide

This file defines how presentations for `phaseIdCrossCorrelation` should be planned and built.

Use this guidance whenever preparing a `.pptx` for:

- lab meetings
- conference talks
- sponsor or project reviews
- benchmark-result walkthroughs
- workflow or GUI demonstrations

## 1. Required Starting Point

Always start from the repository template:

- `ppt_template.pptx`

This template is the visual authority for future slide work in this repository.

Do not improvise a new slide system when building a deck for this project unless explicitly requested.

## 2. Required Skills For Actual PPTX Work

When an agent is asked to actually create or modify a presentation, it must invoke:

- [$slides](/Users/kvman/.codex/skills/slides/SKILL.md)
- [$scientific-slide-house-style](/Users/kvman/.codex/skills/scientific-slide-house-style/SKILL.md)

Use additional skills only when the task needs them:

- [$imagegen](/Users/kvman/.codex/skills/.system/imagegen/SKILL.md) for bitmap assets, conceptual illustrations, or cleaned visual backgrounds

The expected deck-building path is:

1. study `ppt_template.pptx`
2. follow the repository house style below
3. build editable slides with the required slide skills
4. validate the deck visually before delivery

## 3. Core Slide Layout Doctrine

Every content slide should follow this structural pattern:

- top ribbon with the slide title
- bottom ribbon with the slide takeaway
- narrow text rail on the left for grouped bullet points
- large right-side area reserved for graphics, plots, maps, figures, tables, or visual comparisons

This means the slide should read in this order:

1. title in the top ribbon
2. short grouped bullet text on the left
3. dominant scientific visual on the right
4. one-sentence takeaway in the bottom ribbon

The visual should carry most of the slide. Text should support it, not compete with it.

## 4. Visual Style Rules

- Use `Arial` throughout.
- Use a black top ribbon for the title.
- Use a black bottom ribbon for the takeaway.
- Use white or very light neutral content backgrounds between the ribbons.
- Keep the slide graphics-first.
- Avoid dense paragraphs.
- Avoid tiny legends and crowded multi-panel layouts unless the sole goal is comparison.
- Keep one main message per slide.

Preferred size hierarchy:

- title ribbon text: strong and prominent
- takeaway ribbon text: clear and slightly smaller than the title
- bullet text: compact but comfortably readable
- figure labels and annotations: large enough to survive projection

## 5. Text Grouping Rules

The left bullet rail is not a transcript area. It is a grouping aid.

Use it for:

- the question being answered
- the method or condition defining the figure
- the comparison axis
- the interpretation cues the audience should keep in mind

Do not use it for:

- long background paragraphs
- methods sections copied from a paper
- raw result dumps
- repeating what is already obvious from the figure

Good bullet grouping pattern:

- context
- what is being compared
- why the visual matters

Each bullet cluster should feel like a caption block, not a document excerpt.

## 6. Figure And Graphics Rules

The main right-side area should contain the primary evidence.

Good candidates:

- predicted phase maps
- IPF-colored maps
- confusion matrices
- benchmark comparison plots
- paired raw vs processed Kikuchi patterns
- workflow diagrams
- curated result panels

When multiple graphics appear together:

- group them only if they answer the same question
- align them tightly
- keep scales, color conventions, and labels consistent
- make the comparison obvious without verbal explanation

For model-comparison slides, prefer:

- one shared reference visual
- multiple directly comparable panels
- a small metric strip or compact comparison table

## 7. Comparison-Slide Rules

Comparison slides are especially important in this project.

When comparing models, methods, scans, or conditions:

- use the same crop, scale, and color mapping across panels
- keep the legend shared when possible
- avoid repeating the same legend in every panel
- place the compared items in one stable grid
- ensure the takeaway states the decision, not just the observation

For example:

- several predicted phase maps may be shown together for model comparison
- only one IPF-colored map is needed if it is identical in meaning across models

## 8. Scientific Content Rules

Slides must preserve scientific meaning.

- Keep phase names, labels, and notation consistent with the repository outputs.
- Do not relabel Cu/Ni/Al or EBSD fields casually.
- Preserve confidence, split, and benchmark terminology from the code and reports.
- When showing performance, distinguish clearly between validation and held-out test metrics.
- When showing inference maps, make it obvious whether the slide demonstrates qualitative agreement, disagreement, uncertainty, or artifact behavior.

## 9. Workflow For Future Deck Creation

When creating a deck for this repo:

1. identify the story arc before laying out slides
2. collect the exact report artifacts, figures, and tables
3. decide slide-by-slide takeaways
4. map each takeaway to one dominant visual
5. place only supporting grouped bullets on the left rail
6. build in the repository house style using the required skills
7. validate for overflow, overlap, font substitutions, and visual consistency

## 10. Validation Expectations

Before considering a deck complete:

- render the slides to images
- inspect for overflow and out-of-bounds elements
- verify fonts remain consistent with the intended style
- confirm the takeaway ribbon states the actual conclusion of the slide
- confirm the visual remains dominant over the text

## 11. Repository-Specific Instruction

For this repository, `ppt_template.pptx` is the standing guide for future presentation work.

If a future task says "make slides" or "prepare a presentation" for this project, the agent should:

- begin from `ppt_template.pptx`
- follow this guide
- invoke [$slides](/Users/kvman/.codex/skills/slides/SKILL.md)
- invoke [$scientific-slide-house-style](/Users/kvman/.codex/skills/scientific-slide-house-style/SKILL.md)

Only deviate from this style if the user explicitly requests a different presentation format.
