# Research Boundaries and Dependencies / 研究边界

Last decision: 2026-10-19

## Three independent tracks
1. **Five-paper coordinated-manipulation reproduction**: remains an active research priority in the separate `dual-arm-coordinated-manipulation-reproduction` repository.
2. **Dual-FR3 main research project**: **FROZEN** for new development. Do not add new FR3 functionality, FR3 RL/force-control integration or FR3 scenario porting until the five paper reproductions have been completed and the user explicitly reauthorizes FR3 development.
3. **Four open-source benchmark reproductions**: independent learning/portfolio work; may start with ManiSkill as the primary manipulation + RL goal. They are NOT dependent on finishing the five papers and must keep environments/logs/resources isolated.

## Agent guardrail
- Discussion/comparative notes on possible transfer to FR3 are allowed; actually changing FR3 repositories or running new FR3 experiments is not allowed in this phase.
- Keep pinned upstream commits, datasets and simulation environments separate.
- Understand methods -> reproduce official benchmark -> perform controlled ablation -> produce quantitative and interview evidence.
- Do not start full training without a declared compute budget and user approval.
- User owns research understanding, algorithm interpretation, experiment acceptance and oral interview capability; Codex may implement bounded tasks, write tests and maintain reproducibility.
- No result is marked complete without run artifacts and verified claims.
