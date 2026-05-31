# LMS Integration (Phase 4)

Taver exposes a **C API** suitable for wrapping in Node, Python, Java, or C# and calling from an LMS plugin.

## Integration pattern

1. Host app loads `taver.dll` and model assets.
2. Lesson supplies **target text** (`taver_set_target_text`).
3. Learner records audio in the LMS (or uploads WAV).
4. Host calls `taver_score_wav` or `taver_score_samples`.
5. Map `TaverResult` to xAPI or SCORM fields:
   - `overall_score` → mastery
   - `phonemes[]` → granular feedback
   - `words[]` → per-word grades

## Suggested xAPI extensions

```json
{
  "result": {
    "score": { "scaled": 0.87, "raw": 87, "min": 0, "max": 100 },
    "extensions": {
      "https://taver.dev/phonemes": [ "... per-phoneme detail ..." ]
    }
  }
}
```

## Offline-first classrooms

Bundle model + espeak on lab machines; no internet required after install.
