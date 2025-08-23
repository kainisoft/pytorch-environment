# Features Directory

This directory contains all feature-specific projects and learning modules. Each subdirectory represents a separate feature or learning project.

## Current Features

- **udemy/**: Udemy course materials and projects
- **chatbot/**: Chatbot development projects

## Directory Structure

Each feature directory should follow this recommended structure:

```
feature-name/
├── README.md          # Feature-specific documentation
├── notebooks/         # Jupyter notebooks for this feature
├── data/             # Feature-specific datasets
├── models/           # Trained models and checkpoints
├── scripts/          # Python scripts and utilities
├── configs/          # Configuration files
└── utils/            # Helper functions and utilities
```

## Adding New Features

To add a new feature:

1. Create a new directory under `features/`
2. Follow the recommended directory structure above
3. Add a README.md explaining the feature
4. No Docker configuration changes needed - the entire `features/` directory is automatically mounted

## Docker Integration

The entire `features/` directory is automatically mounted to `/workspace/features/` in the Docker container. This means:

- All feature directories are immediately available in Jupyter Lab
- No need to modify `docker-compose.yml` when adding new features
- Changes made in the container are reflected on the host system
- New features can be added without restarting the container

## Best Practices

1. **Keep features isolated**: Each feature should be self-contained
2. **Use relative imports**: When importing between features, use relative paths
3. **Document dependencies**: List any feature-specific requirements in the feature's README
4. **Shared resources**: Use the parent-level directories (`/workspace/data`, `/workspace/models`) for resources shared across features
