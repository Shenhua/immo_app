# ROLE: Senior Technical Architect & Engineering Lead

# CONTEXT
I am onboarding you to this existing codebase. Your goal is to perform a "Deep Dive" analysis to understand the project's intent, current state, and quality.
My ultimate goal is to transition this project from its current state to a **production-grade, stable, and highly modular system**. Do NOT generate new code yet. Your output will be a comprehensive audit and a transformation roadmap.

# INSTRUCTIONS

## Phase 1: Context & Intent Analysis
1. Scan the file structure, configuration files, and dependencies to determine the tech stack.
2. Read the entry points and main logic files to infer the **Business Intent**: What problem does this application solve?
3. Map the **System Architecture**: How do the components talk to each other?

## Phase 2: Technical Audit (The "Etonement" Report)
Analyze the code specifically for "Health," "Logic," and "Maintainability":
1. **Logic Errors & Bugs:** Identify specific logic flaws, race conditions, or unhandled edge cases.
2. **Structural Integrity:** Evaluate if the folder structure follows SOLID principles. Is it modular?
3. **Dependency Health:** Check `package.json` / `requirements.txt`. Are libraries outdated, deprecated, or overkill for the task?
4. **Type Safety:** (If applicable) Assess the use of strict typing (TypeScript types, Python type hints). Is the code "loose" and prone to runtime errors?
5. **Dead Code:** Identify unused files, variables, or "zombie" functions that should be deleted.

## Phase 3: Strategic Planning & Refactoring
Based on your audit, propose a plan to **refactor** the codebase into a solid, scalable structure.
1. **Refactoring Strategy:** Propose a new directory structure or design pattern (e.g., Clean Architecture, MVC, Hexagonal).
2. **Standardization Plan:** Define the rules for logging, testing, error handling, and **Type definitions**.

# DELIVERABLES
Please output your response in the following Markdown format:

## 1. Project Identity Card
* **Project Purpose:** (One sentence summary)
* **Current Stack:** (Languages, frameworks, key libraries)
* **Architecture Overview:** (Current data flow)

## 2. Technical Health Report
* **Critical Issues:** [Bugs/Logic errors found]
* **Dependency Risks:** [Outdated or dangerous libraries]
* **"Spaghetti Code" Areas:** [Parts of the code that are brittle]
* **Missing Standards:** [e.g., "No Type definitions," "No tests"]

## 3. The Refactoring Blueprint (The "Solid" Plan)
* **Proposed Architecture:** (Describe the ideal structure)
* **Directory Structure:** (Visualize the ideal folder tree)
* **Tech Stack Updates:** (Libraries to add, remove, or update)
* **Best Practices to Implement:**
    * **Logging Standard:** (Library/pattern)
    * **Testing Strategy:** (Unit vs Integration)
    * **Type Safety:** (Strictness level)

## 4. Master Development Plan
Create a step-by-step Roadmap. Break this down into a **Task List**.
* [ ] **Phase 0 (Cleanup):** Delete dead code and fix dependencies.
* [ ] **Phase 1 (Stabilization):** Fix critical bugs and security risks.
* [ ] **Phase 2 (Refactoring):** Move code to the new structure.
* [ ] **Phase 3 (Standardization):** Add logging, tests, and types.

# CONSTRAINT
If you are unable to read specific files or need more context to understand a dependency, ask me clarifying questions before generating the final report.