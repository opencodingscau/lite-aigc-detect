# Matched modern-generator protocol (Pilot C)

Bias controls (all required for “matched” claims):

- content / semantic pairing (e.g. real-conditioned fake)
- same target size, codec, colorspace
- strip metadata; identical resize/crop/save pipeline

Split axes independently:

1. **Generator shift** — overlapping semantics, different generators  
2. **Content shift** — fixed generator, different semantics  

See `../configs/pilot_c_matched_gen.yaml` for generator groups.
