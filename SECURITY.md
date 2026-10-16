# Security policy

## Supported versions

Security fixes are made for the latest release of `beam-model`. Please
upgrade (`pip install -U beam-model`) before reporting.

## Reporting a vulnerability

Please do **not** open a public issue. Report it privately through GitHub:
open the repository's **Security** tab and choose **Report a vulnerability**
(<https://github.com/MuteJester/BEAM/security/advisories/new>).

Include what you found, how to reproduce it, and the BEAM version and
platform. You will get an acknowledgement within a week, and we will agree
on a fix and a disclosure date with you before anything is made public.

Examples of what to report: a model file that makes BEAM crash, read or
write outside its buffers, or run arbitrary code when loaded; or a way to
make `BEAM.load_pretrained` accept a file that does not match its published
checksum.
