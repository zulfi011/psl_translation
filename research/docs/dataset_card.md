# Dataset card — PSL landmarks (draft)

**Status:** template, fill in before release.

## Summary
- Signs: 24 PSL signs (+ `noclass`), listed in `data/metadata/clips.csv`
- Signers: N = 5 (anonymous IDs 1–5)
- Clips: _TBD_ total, _TBD_ per sign per signer
- Released form: MediaPipe Holistic landmarks (hands + pose, 258 values/frame), full length; **no video**

## Collection
- Recording device, resolution, frame rate: _TBD_
- Environment (indoor, background, lighting): _TBD_
- Signers' background (native / learner, handedness), only where consented: _TBD_
- Consent: written consent obtained for release of landmark data: _yes/no, date_

## Splits
- `splits/loso_nextval.json`: leave-one-signer-out, 5 folds
- `splits/holdout_v1.json`: original FYP protocol (comparison only)

## Intended use / limitations
- Research on isolated PSL sign recognition.
- Small number of signers; not representative of all PSL users or regional variants.
- Not for high-stakes communication without human verification.

## Licence and citation
- CC BY 4.0 _(or CC BY-NC 4.0)_, DOI: _TBD_
