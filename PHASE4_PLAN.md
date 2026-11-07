# Phase 4 plan: model design and application design

Phase 4 is a **design phase**. No implementation is required for the deliverable (implementation is Phase 5's deliverable, but it may start now).

## Brief (from the course)

- Pick the algorithm that best fits the problem, based on the nature of the data, its size and the pilot results. It does not have to be the best performer; weigh pros, cons and feasibility in the utility application.
- Design the model to fit the data and objective: loss function, any custom feature weighting, any changes to the default design.
- Design the utility application that uses the model:
  - takes normal human input (for us: a comment as a text string) and returns the predictions;
  - must not assume the model is on the same machine: client/server, with the model behind an API;
  - design covers architecture, human interaction flow and data flow.
- **Deliverable:** a design document with the model design and architecture (with justifications) and the application design and architecture.

## Phase 3 feedback to answer (graded Excellent)

1. Validation must use k-fold cross-validation, not a separate validation split.
2. The MLP choice is agreed.
3. Plan how to choose the MLP's hyperparameters. Use focal loss instead of BCE because labels are scarce; also vary the activation function, hidden layer sizes, etc.

The design document should say how each point is handled.

## Model design decisions (agreed 2026-11-07)

| Topic | Decision |
| --- | --- |
| Algorithm | MLP (Phase 3 choice, agreed by the prof) |
| Framework | PyTorch, because sklearn's `MLPClassifier` cannot use a custom loss |
| Validation | Merge the validation split into training; stratified 5-fold CV on train + validation, folds grouped by article; test stays the only holdout |
| Outputs | Try one 6-output multi-label network against six single-label networks; keep the one that wins in CV |
| Search | Staged random search on a ~300k-row sample: stage 1 loss and architecture, stage 2 activation, regularisation and learning rate; refit the best setting on all rows |
| Decision cut | Chosen per label from out-of-fold predictions |

**Search space:** loss (BCE, weighted BCE, focal with γ in {1, 2, 3} and α), activation (ReLU, tanh, sigmoid, leaky ReLU), 1–3 hidden layers of 64–512 units, dropout, L2 weight decay, learning rate, batch size, optimiser, early-stopping patience, text features (Word2Vec, PV-DBOW, or both).

**To prepare:** `article_id` is not in the cleaned parquet; join it back from the raw files by `id` to build article-grouped folds.

## Application design points to cover

- Input is the raw comment text, so the server must run the whole Phase 2 pipeline on it: text cleaning, Word2Vec / Doc2Vec inference, the 12 text statistics, and scaling with the saved training parameters.
- The tabular features `is_reply` and the publication one-hot are not part of a plain text input: the client could ask for them as optional fields, or the server uses defaults ("other" publication, not a reply).
- Output: a score and a yes/no per label, using the cuts chosen by CV, so a moderator can sort a queue by score.
- Client/server: the model and pipeline sit on a server behind an HTTP API (for example a POST that takes text and returns six scores); the client (web page or moderator queue) only sends text and shows results.
