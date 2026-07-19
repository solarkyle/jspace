# Security Policy

jspace is a research codebase: analysis scripts, frozen classifiers, and
published data. It ships no server, service, or install hook. The interactive
demo is a static GitHub Pages site that loads pre-computed JSON — no backend,
no data collection.

## Reporting a vulnerability

If you find a security issue (e.g., in the demo page, a dependency, or
anything that could affect people running the reproduction script), please
report it privately:

- **GitHub**: use [private vulnerability reporting](https://github.com/solarkyle/jspace/security/advisories/new), or
- **Email**: fintechkyle@gmail.com

Please don't open a public issue for security reports. You'll get a response
within a few days; fixes for anything real will be released promptly and
credited to you unless you prefer otherwise.

## Scope notes

- The frozen classifiers in `campaign/frozen/` are SHA-256 hashed in the
  preregistration files; verifying those hashes (as `reproduce_mini.py` does)
  protects against tampered artifacts.
- Data files are downloaded from the public Hugging Face dataset over HTTPS.
