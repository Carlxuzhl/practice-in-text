# Third-party software, models and data

Nothing listed here is redistributed in this repository; the setup scripts fetch it from its source.

| What | Source | Licence | How it is used |
|---|---|---|---|
| MobileGym (code) | https://github.com/Purewhiter/mobilegym, commit `5fbd3f34be0e9780ae4b1e9d5089963ddbd05d38` | Apache-2.0 | The simulated apps, the task templates and the judge; cloned and built by `scripts/setup_sim.sh` |
| MobileGym (data) | same repository | CC BY-NC 4.0 | Academic, non-commercial use only. Screenshots, text views and labels generated from MobileGym are covered by it |
| lucide-react icons | installed by MobileGym's `npm ci` | ISC | `textview/build_icon_table.mjs` builds a shape-to-name table from the icons' geometry at setup |
| Qwen3-8B, Qwen3-VL-4B-Instruct, Qwen3-VL-8B-Instruct | Hugging Face Hub, revisions in `configs/models.json` | see each model card | Policy and screen readers; downloaded by `scripts/download_models.py` |
| InternVL3-8B-hf | Hugging Face Hub, revision in `configs/models.json` | see its model card | Optional reader from another model family |
| Python and Node packages | PyPI, npm | their own licences | `requirements/*.txt`; MobileGym's `package-lock.json` |
