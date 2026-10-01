# Full Offline Wikipedia with Kiwix

This guide sets up a **complete offline English Wikipedia** using Kiwix.

## Editions

Kiwix publishes English Wikipedia as ZIM files at https://download.kiwix.org/zim/wikipedia/, refreshed every few months. File names carry the dump month (e.g. `wikipedia_en_all_nopic_2026-06.zim`) and old ones are removed from the mirror, so never hardcode a name.

| Edition | Contents | Size (2026) |
| --- | --- | --- |
| `wikipedia_en_all_nopic` | all articles, no images | ~49 GB **(recommended)** |
| `wikipedia_en_all_mini` | lead section + infobox only | ~13 GB |
| `wikipedia_en_all_maxi` | all articles with images | ~119 GB |

Images don't help the assistant, so `nopic` gives the same text as `maxi` for less than half the disk.

## Quick Setup

```bash
./scripts/setup_full_offline_wikipedia.sh                  # asks which edition (default nopic)
./scripts/setup_full_offline_wikipedia.sh --variant mini   # small-disk option
./scripts/setup_full_offline_wikipedia.sh --dry-run        # just print the newest URL
```

The script:
- installs `kiwix-serve` (latest kiwix-tools release) to `~/.local/bin/` if missing
- lists the mirror and picks the newest file for the chosen edition
- checks free disk space and downloads to `~/wikipedia-offline/` with `wget -c` (re-run to resume)

Set `WIKI_DIR` to download somewhere else.

**Manual download:** pick the newest `wikipedia_en_all_nopic_*.zim` from the mirror listing and `wget -c` it into `~/wikipedia-offline/`.

## Use with the assistant

```bash
./run.sh
```

`run.sh` uses the newest **complete** ZIM in `~/wikipedia-offline/` (also `~/Downloads/`, `/data/wikipedia/`, `/var/lib/kiwix/`). A file that is still downloading is detected from its header and skipped. To choose a file explicitly:

```bash
./run.sh --zim /path/to/wikipedia_en_all_nopic_2026-06.zim
export WIKI_ZIM=/path/to/wikipedia_en_all_nopic_2026-06.zim   # same, via environment
```

## Browse Wikipedia

```bash
./scripts/start_offline_rag.sh     # serves the same ZIM at http://localhost:8080
```

The assistant's own auto-started server listens on 127.0.0.1 only; `start_offline_rag.sh` listens on all interfaces so other devices on your network can browse.

## Keeping Kiwix Running (systemd)

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/kiwix.service <<EOF
[Unit]
Description=Kiwix Wikipedia Server
After=network.target

[Service]
Type=simple
ExecStart=$HOME/.local/bin/kiwix-serve --port=8080 $HOME/wikipedia-offline/wikipedia_en_all_nopic_2026-06.zim
Restart=on-failure

[Install]
WantedBy=default.target
EOF

systemctl --user enable --now kiwix
```

Update the file name in `ExecStart` when you download a newer dump. With the service running, `./run.sh` connects to it instead of starting its own server.

## System Requirements

- **Disk**: ~50 GB for `nopic` (plus models)
- **Kiwix server**: ~100 MB RAM
- **Download time**: a few hours, depending on your connection

## Troubleshooting

**kiwix-serve not found:** the setup script installs it to `~/.local/bin/`. The assistant looks there automatically; for your shell, add it to `PATH`:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

**"Skipping incomplete ZIM":** the download hasn't finished. Re-run the setup script to resume it.
