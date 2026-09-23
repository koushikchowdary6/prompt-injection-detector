# Security Model

## What this component does

The detector is a **pre-model input signal** for LLM applications. It combines transparent heuristics with a lightweight text classifier to identify inputs that resemble prompt-injection or jailbreak attempts.

It should be treated as one layer in a larger security architecture, not as an authorization mechanism.

## Threat model

The attacker controls text that may be passed to an LLM. Their goal may be to override application instructions, reveal hidden context, change the model's role, bypass restrictions, or influence downstream tool use.

The detector evaluates that text before the application sends it to the model.

## Security properties

### Explainable high-signal rules

Regex heuristics catch recognizable attack patterns such as attempts to ignore prior instructions or reveal a system prompt. These rules are fast and auditable but cannot generalize to every paraphrase.

### Statistical generalization

The TF-IDF + logistic-regression layer can score phrasings that do not exactly match a heuristic. Because the bundled training set is small, this model demonstrates the architecture rather than establishing production-grade detection accuracy.

### No claim of complete prevention

A low score must never be interpreted as proof that input is safe. Attackers can obfuscate instructions, distribute them across context, use indirect prompt injection, or exploit application-specific logic that an input-only classifier cannot observe.

## Recommended defense in depth

A production LLM application should combine an input signal like this with:

1. least-privilege tools and credentials;
2. explicit authorization checks outside the model;
3. separation between trusted instructions and untrusted retrieved content;
4. validation of tool arguments and sensitive outputs;
5. confirmation before irreversible actions;
6. logging and evaluation of attempted attacks;
7. adversarial tests tailored to the application's actual workflows.

## Known evaluation limitations

The repository's bundled dataset contains 40 hand-written examples. Cross-validation results describe performance on that illustrative set only. They should not be generalized to unseen production traffic or treated as a benchmark against other guardrails.

Useful next evaluation work includes indirect injection embedded in retrieved documents, multilingual and encoded attacks, multi-turn attacks, benign inputs containing security terminology, and application-specific false-positive analysis.
