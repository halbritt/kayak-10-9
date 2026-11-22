# Gluing the 1:8 model with small bar clamps

[Download the H2S collar plate](https://github.com/halbritt/kayak-10-9/releases/download/glue-collars-v1/kayak-10-9-scale-1to8-h2s-glue-collars.3mf) · [Placement diagram](https://github.com/halbritt/kayak-10-9/releases/download/glue-collars-v1/clamp-placement.png)

Print the plate once at 100% in its supplied orientation. It contains two removable halves for each of four collars. The broad faces lie flat on the bed. The dots on the upper face identify each collar.

| Dots | Section to grip | Joint |
| --- | --- | --- |
| 1 | Bow | Forward |
| 2 | Front of cockpit section | Forward |
| 3 | Back of cockpit section | Aft |
| 4 | Stern | Aft |

1. Dry-fit the sections and their alignment keys. Put each collar around its section with its smooth face **3 mm back from the seam** and its dotted face pointing away from the seam. The flatter inside curve follows the deck. The two halves meet on the deck and keel centerlines.
2. Loop one cable tie around each collar in its outer groove. Use a tie at least 350 mm long and up to 4.8 mm wide, or link two shorter ties. Set the buckle on top, clear of the clamp jaws. Cinch until the collar resists sliding by hand. For a slick surface, place thin rubber strips against the hull sides and repeat the dry fit.
3. Apply adhesive to the joint. Put one small bar clamp on the port side and another on starboard, with both jaws pressing on the flat outer faces of the paired collars. The opening between jaw faces is 26 mm with the seam closed. Bring the clamps up evenly, just until the bulkheads meet. If a collar creeps, release the clamps and improve its grip. The keys keep the sections aligned.
4. Wipe squeeze-out through the 6 mm gap. After the adhesive cures, release the clamps, cut the ties and lift the collar halves off sideways.

The collars were checked against the v0.1.2 release STLs. The v0.1.1 release uses the same scale-model archive. Fit and holding force on a physical print depend on surface finish and print tolerances.

To regenerate, extract the model STLs to `build/model_1to8_h2s/` or run `cad/build_model_1to8_h2s.py`. Install `manifold3d` alongside the CAD dependencies and run `python3 cad/build_glue_collars.py`. The plate is written to `build/glue_collars/`.
