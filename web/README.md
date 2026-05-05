# Codec-SUPERB Website

This React app builds the official Codec-SUPERB project page and the codec-superb-tiny leaderboard.

## Available Scripts

Run commands from the `web/` directory.

### `npm start`

Runs the app in development mode at [http://localhost:3000](http://localhost:3000).

### `npm run build`

Builds the app for production to the `build/` folder. The output is minified and filenames include content hashes.

## Updating Results

From the repository root, run:

```bash
python3 scripts/update_leaderboard.py
```

The script reads result JSON files from `results/codec-superb-tiny/`, excludes `llmcodec_abl_*` variants from the published leaderboard, and rewrites `web/src/results/data.js`.

## Deployment Notes

The app uses the `homepage` setting in `package.json` for GitHub Pages builds. Keep public assets such as `Overview.png` under `web/public/`.
