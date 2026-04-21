# Contributing to Tox-Agent

Thank you for contributing to the Tox-Agent project. As a member of the NEU Bio Research Team, your contributions help advance our molecular toxicity analysis capabilities. We follow strict engineering standards to ensure our research tools remain reliable, secure, and scientifically accurate.

## Branching Strategy

We utilize a simplified Gitflow approach to manage research and production stability:

- `main`: Contains the stable, production-ready code currently deployed to Google Cloud.
- `develop`: The primary integration branch for new features and research experiments.
- `feature/description`: Created from `develop` for specific tasks, for example `feature/gatv2-optimization`.
- `fix/description`: Used for resolving bugs or data-processing errors.

## Commit Convention

We use Conventional Commits to keep our history readable and to automate changelog generation.

```text
<type>(<scope>): <description>

[optional body]
```

### Common Types

- `feat`: A new feature for the API or AI engine, for example `feat(gnn): add attention visualization`
- `fix`: A bug fix, for example `fix(parser): handle invalid SMILES strings`
- `docs`: Documentation changes, for example `docs(api): update OpenAPI spec`
- `research`: Changes to model hyperparameters or datasets that do not affect code logic
- `refactor`: Code changes that neither fix a bug nor add a feature

## Pull Request Process

1. Sync: Ensure your branch is up to date with `develop`.
2. Test: Run `npm test` and verify the Python inference engine executes correctly.
3. Template: Fill out the PR template, specifically noting whether the change affects model accuracy or data privacy (RLS).
4. Review: At least one peer review from a core member is required: Teddy, Nhật Minh, or Nghĩa Nguyễn.

## Code Review Standards

Our reviewers focus on four critical pillars:

- **Scientific Integrity**: Does the change negatively impact the GATv2 model's precision or interpretability?
- **Security**: Does the code respect PostgreSQL Row-Level Security? Ensure there is no data leakage between research profiles.
- **Maintainability**: Is the MVC pattern preserved? Avoid logic leaks from the model into the view.
- **Performance**: Ensure molecular graph construction does not cause memory bottlenecks during high-throughput screening.

## Definition of Done

A task is considered done only when all of the following are true:

- [ ] Code passes all linting and unit tests
- [ ] `openapi.yaml` is updated for any API changes
- [ ] Documentation in `RESEARCH.md` or `README.md` is updated
- [ ] For architectural shifts, an ADR has been created in `docs/adr/`
- [ ] Firebase Security Rules and RLS policies have been verified if the schema changed

## Reporting Issues

- **Security vulnerabilities**: Do not open a public issue. Contact the team immediately at `security@neu.edu.vn`.
- **Bugs and feature requests**: Use the provided GitHub issue templates so the team has enough context to reproduce the problem or evaluate the request.
