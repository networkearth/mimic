import click

@click.group()
def cli():
    pass

@cli.group()
def log_odds():
    pass

@log_odds.command()
@click.argument("input")
def run(input):
    print(f"Running log odds on {input}")