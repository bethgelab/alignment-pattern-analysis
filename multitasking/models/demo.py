"""Command-line interface to quickly test model inference."""

import random

import click
import questionary
import torch

from multitasking.models import (
    FAVORITE_MODELS,
    build_model,
    get_default_layers,
    list_models,
)


@click.command()
def main():
    """Command-line interface to quickly test model inference."""
    num_models = len(list_models())
    click.echo(
        f"In total, there are {num_models} models available. This script only uses our "
        f"favorite {len(FAVORITE_MODELS)} models."
    )

    providers = sorted(set(model.split("/")[0] for model in FAVORITE_MODELS))
    provider = questionary.select(
        "Which provider do you want to use?",
        choices=providers,
    ).ask()

    model_names = [name for name in FAVORITE_MODELS if name.startswith(provider)]
    model_name = questionary.select(
        "Which model do you want to use?",
        choices=model_names,
    ).ask()

    click.echo(f"Building model '{model_name}'...")
    model = build_model(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    click.echo(f"Using device: {device}")
    model.to(device)

    click.echo(f"The model provides {len(model.layers)} layer(s).")

    default_layers = get_default_layers(model_name)
    if default_layers is not None:
        click.echo(f"The model provides default layers: {default_layers}")
        layers = default_layers
    else:
        layers = random.sample(model.layers, 3)
        click.echo(
            f"The model does not provide default layers. Let's choose 3 random layers: "
            f"{', '.join(layers)}"
        )

    video = torch.randint(0, 256, (1, 3, 10, 256, 256), dtype=torch.uint8)
    click.echo(f"Input video shape (BCTHW): {tuple(video.shape)}")

    model.eval()
    with torch.no_grad():
        output = model.extract_video_features(video, layers)

    click.echo("Feature shapes:")
    for layer, layer_output in output.items():
        click.echo(f"- {layer}: {layer_output.shape}")

    click.echo("Done.")


if __name__ == "__main__":
    main()
