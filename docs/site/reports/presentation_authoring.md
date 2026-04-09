# Presentation Authoring

This page defines the repository house style for future `.pptx` creation.

Use it whenever presentation work is requested for:

- benchmark reporting
- conference or lab-meeting talks
- inference-map review decks
- workflow and GUI demonstrations

## Mandatory Starting Point

Start from:

- `ppt_template.pptx`

Treat this template as the standing visual authority for this repository.

## Skills To Invoke

When actually building or modifying a deck, agents must invoke:

- [$slides](/Users/kvman/.codex/skills/slides/SKILL.md)
- [$scientific-slide-house-style](/Users/kvman/.codex/skills/scientific-slide-house-style/SKILL.md)

Optional supporting skill:

- [$imagegen](/Users/kvman/.codex/skills/.system/imagegen/SKILL.md)

## Slide Structure

Content slides should use the following structure:

1. top ribbon for the title
2. bottom ribbon for the takeaway
3. left-side grouped bullet rail
4. large right-side figure area

The figure area should dominate the slide. The bullet rail should behave like a caption rail, not like prose.

## Visual Doctrine

- Arial only
- black title ribbon
- black takeaway ribbon
- white or near-white content area
- one slide, one message
- graphics first
- short grouped text only

## Text Grouping Rules

Use the left rail for:

- context
- comparison axis
- what the audience should notice

Avoid:

- paragraph-heavy narrative
- duplicated figure description
- dense method text

## Comparison Rules

For multi-model or multi-condition comparison slides:

- keep panel scales aligned
- share legends whenever possible
- use one stable layout grid
- put the conclusion in the bottom takeaway ribbon

For this repository specifically:

- multiple predicted phase maps can be compared side by side
- only one IPF-colored map is usually needed if it is identical in interpretation across models

## Scientific Rules

- preserve repository phase naming and notation
- distinguish validation from held-out test metrics
- do not weaken scientific precision for visual convenience
- keep confidence, provenance, and benchmark wording consistent with generated artifacts

## Required Working Process

1. inspect `ppt_template.pptx`
2. define the takeaway for each slide
3. select the dominant visual first
4. add only grouped supporting bullets
5. build the deck with the required skills
6. validate rendering, bounds, and font consistency

## Validation

Before delivery:

- render slide previews
- check overflow and out-of-bounds elements
- confirm font consistency
- confirm the takeaway matches the actual evidence

## Repository Rule

If a future task asks for a presentation in this repository, the default behavior should be:

- use `ppt_template.pptx`
- follow this page
- invoke [$slides](/Users/kvman/.codex/skills/slides/SKILL.md)
- invoke [$scientific-slide-house-style](/Users/kvman/.codex/skills/scientific-slide-house-style/SKILL.md)

Only deviate if the user explicitly asks for another presentation style.
