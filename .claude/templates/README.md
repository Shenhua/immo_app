# Standard Documentation Framework
This directory contains templates for the standard engineering documentation framework. Having a structured approach to documentation ensures that a project is thoughtfully planned before building, consistently tracked during development, and easily understood by new contributors after it is built.

## Project Documentation Folder Structure
A golden standard for any project repository is to maintain a `docs/` folder structured as follows:

```text
/project-root
 ├── /docs
 │   ├── /01_requirements        # PRDs (What are we building and why?)
 │   ├── /02_design              # TDDs (How are we building it? Data schemas, API contracts)
 │   ├── /03_architecture        # ADRs (Why did we choose this tech/pattern?)
 │   ├── /04_as_built            # SADs (How does the current system actually work?)
 │   └── /05_guides              # Runbooks, Setup instructions, User manuals
 ├── README.md                   # The entry point (Quick start, local dev setup)
 └── INSTRUCTIONS.md             # AI Agent / Dev constraints (e.g., Code rules, stacks)
```

## Which Document Do I Need?

### 1. Before You Code: Product Requirements Document (PRD)
**File Template:** `01_PRD_Template.md`
**Intention:** Defines the **"What"** and **"Why"**. It is written to align stakeholders (Product, Business, Engineering) on what the specific feature or application is supposed to do, who it is for, and what the success criteria are. It focuses on User Journeys and Business Logic, independent of the tech stack.

### 2. Before You Code: Technical Design Document (TDD)
**File Template:** `02_TDD_Template.md`
**Intention:** Defines the **"How"**. Translates the PRD into an engineering blueprint. It maps out the database schemas, API contracts, system architecture diagrams, and edge-case handling. A developer should be able to read this document and immediately start writing the exact structure of the code.

### 3. During Development: Architecture Decision Record (ADR)
**File Template:** `03_ADR_Template.md`
**Intention:** Defines the **"Why this way"**. Whenever your team makes a significant technical choice (e.g., "Switching from REST to GraphQL", or "Using Genetic Algorithms instead of Brute Force"), you write an ADR. It creates a historical log so future engineers understand *why* a particular decision was made given the constraints at the time.

### 4. After Development: System Architecture Document (SAD)
**File Template:** `04_SAD_Template.md`
**Intention:** Defines the **"What is currently built"**. This is an *As-Built* document (similar to the one we generated earlier). It is a living document that explains how the existing codebase operates, what files do what, and where the current defects/technical debt lie. It is primarily used for onboarding new developers to an existing codebase.
