# XYLA — AI Smart Mirror Simulator

> The mirror that sees you, talks to you, and has your back.

![Xyla](https://img.shields.io/badge/status-building-00f5ff?style=flat-square) ![AI](https://img.shields.io/badge/AI-Claude%20Vision-ff00aa?style=flat-square) ![License](https://img.shields.io/badge/license-MIT-ffd700?style=flat-square)

## What is Xyla?

Xyla is an AI-powered smart mirror that combines:
- 🪞 **Live mirror feed** via webcam (desktop sim) / camera module (hardware)
- 💄 **Makeup analysis** — detects missed spots, blending issues, asymmetry
- 👗 **Outfit review** — checks proportions, colour harmony, styling gaps
- 🗣️ **AI conversation** — powered by Claude, warm and affirming
- 📅 **Calendar** — shows your day's schedule
- 🎵 **Music** — integrated audio playback
- 📸 **Photo capture** — take mirror selfies, send to your phone
- ✨ **Affirmations** — personalised daily affirmations

## Quick Start (Desktop Simulator)

No installation needed. Just open in a browser:

```bash
git clone https://github.com/YOUR_USERNAME/xyla.git
cd xyla
open index.html
# or: python3 -m http.server 3000 → localhost:3000
```

To enable real AI:
1. Get an API key at [console.anthropic.com](https://console.anthropic.com)
2. Enter it in the startup modal
3. Xyla is now fully alive

## Project Structure

```
xyla/
├── index.html          ← Full desktop simulator (start here)
├── README.md
├── hardware/           ← Raspberry Pi setup (coming soon)
│   ├── setup.sh
│   └── requirements.txt
├── src/
│   ├── vision/         ← Computer vision modules (Python)
│   ├── ui/             ← React app for RPi display
│   └── api/            ← Backend services
└── docs/
    └── build-guide.md  ← Hardware build guide
```

## Hardware Build (Prototype - September 2025)

| Component | Cost (ZAR) |
|---|---|
| Raspberry Pi 5 (4GB) | ~R1,400 |
| 10" HDMI Touchscreen | ~R1,500 |
| Two-way acrylic mirror | ~R550 |
| USB wide-angle camera | ~R400 |
| USB microphone | ~R280 |
| Speakers + amp | ~R350 |
| Frame (MDF + hardware) | ~R450 |
| **Total** | **~R4,930** |

## Roadmap

- [x] Desktop simulator with all core features
- [x] Claude API integration (vision + conversation)
- [x] Calendar, music, photo capture
- [ ] Raspberry Pi hardware build
- [ ] Wake word detection ("Hey Xyla")
- [ ] WhatsApp/MMS photo sending
- [ ] On-device ML model for makeup detection
- [ ] Mobile companion app

## Tech Stack

- **Frontend**: HTML/CSS/JS (sim) → React/Electron (hardware)
- **AI**: Anthropic Claude (vision + conversation)
- **Computer Vision**: OpenCV + Claude Vision API
- **Voice**: Web Speech API (sim) → Whisper + Piper TTS (hardware)
- **Hardware**: Raspberry Pi 5, custom mirror frame

## Built by

Thuh Mukangamwi — Building Xyla for as a wedding gift for my mom (September 2026).

---

*"The mirror that sees you."*
