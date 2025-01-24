import os

import typer

from maestro.cli.env import DEFAULT_DISABLE_RECIPE_IMPORTS_WARNINGS_ENV, DISABLE_RECIPE_IMPORTS_WARNINGS_ENV
from maestro.cli.utils import str2bool


def find_training_recipes(app: typer.Typer) -> None:
    from maestro.trainer.models.florence_2.entrypoint import florence_2_app

    app.add_typer(florence_2_app, name="florence2")

    # try:
    #     from maestro.trainer.models.florence_2.entrypoint import florence_2_app
    #
    #     app.add_typer(florence_2_app, name="florence2")
    # except Exception:
    #     _warn_about_recipe_import_error(model_name="Florence 2")


def _warn_about_recipe_import_error(model_name: str) -> None:
    disable_warnings = str2bool(
        os.getenv(
            DISABLE_RECIPE_IMPORTS_WARNINGS_ENV,
            DEFAULT_DISABLE_RECIPE_IMPORTS_WARNINGS_ENV,
        )
    )
    if disable_warnings:
        return None
    warning = typer.style("WARNING", fg=typer.colors.RED, bold=True)
    message = "🚧 " + warning + f" cannot import recipe for {model_name}"
    typer.echo(message)
