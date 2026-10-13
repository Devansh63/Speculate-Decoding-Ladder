# Attribution: logged exception, 13 Oct 2026

## What was checked

Full-history audit of author and committer identity (`b0_05_attribution_audit.sh`).

```
authors:
     51 Devansh Agrawal <56602595+Devansh63@users.noreply.github.com>
     18 Devansh Agrawal <dagraw2@illinois.edu>
      5 Devansh Agrawal <devanshagrawal63@gmail.com>

committers:
     50 Devansh Agrawal <56602595+Devansh63@users.noreply.github.com>
     18 Devansh Agrawal <dagraw2@illinois.edu>
      5 Devansh Agrawal <devanshagrawal63@gmail.com>
      1 GitHub <noreply@github.com>

commits matching an AI/bot pattern in author or committer: 0
0
```

## Finding

The standing constraint is that contributions must not appear as an AI's.
That constraint is **not violated anywhere in this history**.

RULES section 1 additionally requires the literal identity
`Devansh Agrawal <dagraw2@illinois.edu>`. Part of the history predates that
rule and uses the author's other identities -- the GitHub noreply address
and a personal gmail address. Both resolve to the same GitHub account, so
those commits already display as the author in the only place anyone looks.

## Decision

**Logged as an exception; history is not rewritten.** Rewriting would need
`git filter-repo` plus a coordinated force-push, would break every existing
clone and permalink, and would change nothing a reader can see. New commits
continue to use the RULES section 1 identity, which the recent chain already
does.

Re-run the audit script before submission. If the AI/bot count is ever
non-zero, that is a different problem and this exception does not cover it.
