# Contributing

Thanks for helping. This is a small project, so the rules are short.

## Set up and run the tests

```bash
git clone https://github.com/bhargavchintam/screen-vision-utils.git
cd screen-vision-utils
uv run --no-project --with opencv-python-headless --with numpy --with pytest pytest
```

Without uv: `python -m pip install -e ".[dev]"`, then `pytest`.

The tests make their own images with numpy and OpenCV, so they need no files, no network and no Google Cloud account.

## Making a change

- Open an issue first for anything bigger than a small fix.
- Add or update a test for every change in behaviour.
- Keep the core light: `locate` needs only numpy and OpenCV. Anything that needs another package goes behind an optional extra, as `text` does.
- If you change how `locate` draws its output, run `python scripts/make_demo.py` to rebuild `docs/demo.png`.

## Credentials

Never commit a Google Cloud key or any other secret. Keep the key file outside the repository and point `GOOGLE_APPLICATION_CREDENTIALS` at it. The `.gitignore` ignores `*.json` files for this reason.

## License

By contributing, you agree that your contribution is licensed under the MIT License, the same as the project.
