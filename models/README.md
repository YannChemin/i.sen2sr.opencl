# SEN2SRLite model weights

The three SEN2SRLite variants used by *i.sen2sr.opencl*, copied unchanged
from the SEN2SR model repository (CC0 1.0 licence):
<https://huggingface.co/tacofoundation/SEN2SR/tree/main/SEN2SRLite>

| Directory  | Upstream directory                | Bands                      |
| ---------- | --------------------------------- | -------------------------- |
| `main`     | `SEN2SRLite/main`                 | all 10 bands, 10/20 m to 2.5 m |
| `rgbn_x4`  | `SEN2SRLite/NonReference_RGBN_x4` | B02 B03 B04 B08, 10 m to 2.5 m |
| `rswir_x2` | `SEN2SRLite/Reference_RSWIR_x2`   | 20 m bands to 10 m         |

Only the `*.safetensor` weight files are kept (not the upstream example
data, `load.py` or `mlm.json`). `main/sr_model.safetensor` is identical to
`rgbn_x4/model.safetensor`, and `main/f2_*.safetensor` to the two
`rswir_x2` files; `rgbn_x4/hard_constraint.safetensor` differs from
`main/sr_hard_constraint.safetensor`.

SHA-256 checksums (downloaded 2026-09-25):

```
a7d4c8fe617180b43426585ed985d9649a4d8fee655afdc87f1e66b8026caed2  main/f2_hard_constraint.safetensor
7bb734f630329545a806fe907f264122a5feb347dbe698c5529ea95f6cb1edff  main/f2_model.safetensor
cb6370116a248e6d58728a7ec3c3df11fdeae0f1c82ec0e3b424a3734e3f3ba9  main/hard_constraint.safetensor
f45c99af1d08408cb622ba02d1aeadaab2ca964f08991feeb4fd7ffd5197f59d  main/model.safetensor
994a246f1da7aa8df81d95dfcdfa5606a83f9b852636236043d6a67a568fa044  main/sr_hard_constraint.safetensor
479aa796d5068d0b1206118ccbca27bd3223df0214db1a9b31a1e18349ed1c7e  main/sr_model.safetensor
fbad981519066387c413ead1d6af7ef3e0d2947c34147ba90163fc79ae539239  rgbn_x4/hard_constraint.safetensor
479aa796d5068d0b1206118ccbca27bd3223df0214db1a9b31a1e18349ed1c7e  rgbn_x4/model.safetensor
a7d4c8fe617180b43426585ed985d9649a4d8fee655afdc87f1e66b8026caed2  rswir_x2/hard_constraint.safetensor
7bb734f630329545a806fe907f264122a5feb347dbe698c5529ea95f6cb1edff  rswir_x2/model.safetensor
```
