# MaaFramework Skills Analysis

## Summary

`maaframework-skills/` is a compact but well-structured MaaFramework knowledge base centered around one reusable skill: `maaframework`.

It is not a random note collection. Its shape is:

- `README.md`: repository-level overview and installation hint
- `skills/maaframework/SKILL.md`: skill entry, topic map, and navigation
- `skills/maaframework/references/`: condensed reference layer for fast model consumption
- `skills/maaframework/docs/`: fuller source material in both English and Chinese
- `skills/maaframework/GENERATION.md`: provenance metadata

The knowledge base is designed for coding agents or assistants that need to understand MaaFramework quickly and answer implementation questions with less context overhead.

## Coverage

The current corpus covers the main MaaFramework surfaces needed for day-to-day project work:

- Core architecture and terminology
- Pipeline protocol and node lifecycle
- Recognition algorithms
- Action types
- Controller and integration model
- Callback protocol
- `interface.json` / ProjectInterface V2
- Custom recognition and custom action through Agent
- Troubleshooting and debug workflow

Observed file counts:

- `docs/en_us`: 15 markdown files
- `docs/zh_cn`: 15 markdown files
- `references/`: 10 condensed markdown files

Skill metadata states:

- Skill name: `maaframework`
- Source baseline: MaaFramework v5.x
- Generated at: `2026-04-16`

## Information Architecture

This knowledge base uses a sensible two-layer design:

### Layer 1: Source docs

`docs/` contains broader material, including:

- Quick start
- Terms
- Integration
- Callback protocol
- Control methods
- Pipeline protocol
- ProjectInterface
- Build guide
- Standardized interface design
- Troubleshooting
- NodeJS-specific guides

This layer is better when the reader needs completeness, wording details, or protocol edge cases.

### Layer 2: Reference docs

`references/` is the high-value layer for repeated engineering use. It compresses the original docs into smaller task-oriented references:

- `core-architecture.md`
- `core-pipeline.md`
- `core-recognition.md`
- `core-actions.md`
- `core-integration.md`
- `core-controllers.md`
- `core-callbacks.md`
- `core-project-interface.md`
- `advanced-custom-logic.md`
- `advanced-troubleshooting.md`

This is the best layer for an assistant or developer who wants fast retrieval without re-reading all upstream documentation.

## What It Is Good At

This knowledge base is especially strong in the following areas:

### 1. Fast onboarding

`SKILL.md` is small, readable, and routes the reader to the right reference file instead of dumping everything at once.

### 2. Practical MaaFramework development

It covers the exact topics needed to build real projects:

- pipeline JSON
- image/OCR/YOLO-style recognition setup
- custom actions and recognitions
- controller setup
- `interface.json`

### 3. Good match for this repository

This project already relies heavily on the same concepts:

- pipeline-driven task flow in `assets/resource/pipeline/`
- `interface.json` task and option declarations
- Python Agent process in `agent/main.py`
- custom actions in `agent/actions/`
- OCR and neural-network detection in runtime logic

So this knowledge base is directly relevant, not merely adjacent reference material.

### 4. Bilingual source retention

Keeping both `en_us` and `zh_cn` source docs is valuable. It reduces ambiguity when upstream terminology is translated differently across communities.

### 5. Compression without losing the mental model

The `references/` layer preserves structure and engineering intent, not just isolated facts. That makes it much more useful than a flat FAQ.

## Limitations And Gaps

The knowledge base is good, but not complete.

### 1. It is mostly descriptive, not project-opinionated

It explains MaaFramework well, but it does not define team conventions for this repository, such as:

- naming rules for nodes
- option key naming standards
- image asset capture standards for this project
- when to use pure pipeline vs custom Python action
- packaging rules for cross-machine agent paths

### 2. It has weak provenance detail

`GENERATION.md` only records that the content was generated from `docs/` on `2026-04-16`. It does not record:

- exact MaaFramework commit or tag
- transformation pipeline
- whether references were manually edited after generation
- diff strategy for future regeneration

### 3. It does not contain project-specific examples from this repo

For this repository, high-value missing examples would include:

- how `血缘.json` maps to `bloodline_battle.py`
- how `无尽.json` maps to `refresh_endless_affixes.py`
- how `interface.json` options override pipeline fields here

### 4. No quality rubric for resource authoring

The docs mention template quality and 720p assumptions, but this knowledge base does not yet define a local checklist for:

- screenshot scaling
- ROI review
- threshold tuning
- debug image interpretation
- OCR replacement strategy

### 5. No maintenance workflow

There is no document describing when and how this knowledge base should be refreshed as MaaFramework evolves.

## Best-Use Strategy

For future work in this repository, the best retrieval order is:

1. `skills/maaframework/SKILL.md`
2. matching file in `references/`
3. matching fuller doc in `docs/en_us` or `docs/zh_cn`
4. actual code in this repository

Recommended mapping by question type:

- Pipeline syntax or execution behavior: `references/core-pipeline.md`
- OCR, TemplateMatch, YOLO detect behavior: `references/core-recognition.md`
- Click/Swipe/Shell/Command behavior: `references/core-actions.md`
- `interface.json` structure: `references/core-project-interface.md`
- Agent custom logic: `references/advanced-custom-logic.md`
- Debugging and logs: `references/advanced-troubleshooting.md`
- Controller/platform specifics: `references/core-controllers.md`

## Relevance To Current Project

This knowledge base already explains most of the conceptual foundation behind the current repository:

- why the project can stay mostly JSON-driven
- why Python is only used for complex logic
- how `AgentServer.custom_action(...)` fits MaaFramework's Agent model
- how runtime `pipeline_override` works
- how `interface.json` options feed into task execution

In short:

- the current repo is a concrete MaaFramework project
- this skill is a reusable conceptual map for understanding and evolving it

## Persistence Decision

This analysis is persisted in:

- `maaframework-skills/ANALYSIS.md`

Purpose:

- keep the knowledge-base evaluation inside the repository
- make future MaaFramework work faster
- avoid re-deriving the same conclusions in later sessions

## Recommended Next Additions

If this knowledge base is going to be maintained long-term, the highest-value next files would be:

1. `maaframework-skills/PROJECT_MAPPING.md`
   Map this repository's real files to MaaFramework concepts.

2. `maaframework-skills/TEAM_CONVENTIONS.md`
   Define local standards for node naming, options, resources, thresholds, and agent design.

3. `maaframework-skills/UPDATE_POLICY.md`
   Record how to refresh the knowledge base when MaaFramework upstream changes.

4. `maaframework-skills/EXAMPLES_CURRENT_REPO.md`
   Add examples taken from `血缘`, `无尽`, and current custom actions.

## Final Assessment

This is a strong MaaFramework-oriented skill package with a clean structure and high reuse value.

Its biggest advantage is not raw completeness, but the fact that it turns large upstream documentation into a compact operational reference set.

Its biggest weakness is that it stops at framework knowledge and has not yet been localized into repository-specific engineering doctrine.
