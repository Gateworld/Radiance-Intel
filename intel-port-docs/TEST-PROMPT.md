This is a dry run of the unattended runner, with the owner watching. It is not the probe.
Do not read PROMPT.md or UNATTENDED.md, start no probe work, and create no STATE.md, DONE or PAUSED.

Do exactly these four things, then stop:
1. Run `git -C src log -1 --format='%h %ci %s'` and report the Radiance commit.
2. Run `df -h ~/code` and report the free disk.
3. Run `sudo true` once. It should be denied by the permission rules; report what happened.
4. Run `notify "radiance-intel [test]: dry run reached the model; commit <short hash>, <free> free in ~/code"`
   with the real values filled in.

Reply in at most five lines.
