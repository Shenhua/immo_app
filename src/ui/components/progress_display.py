"""
Streamlit progress display component.

Renders real-time progress during strategy search.
"""

import streamlit as st

from src.application.progress import PHASE_LABELS, PHASES_ORDER, SearchPhase, SearchProgress, SearchStats


def render_progress_display(progress: SearchProgress) -> None:
    """
    Render the progress display in Streamlit.

    Shows a multi-step progress indicator with:
    - Completed phases marked with ✅
    - Current phase with progress bar (for evaluation) or spinner
    - Pending phases marked with ⏳

    Args:
        progress: Current search progress state
    """
    st.markdown("### 🔍 Recherche de Stratégies")

    for i, phase in enumerate(PHASES_ORDER, 1):
        phase_label = PHASE_LABELS[phase]

        if i < progress.phase_index:
            # Completed phase
            st.markdown(f"✅ **Étape {i}/{len(PHASES_ORDER)}:** {phase_label}")

        elif i == progress.phase_index:
            # Current phase
            if phase == SearchPhase.EVALUATION and progress.items_total > 0:
                # Show detailed progress bar for evaluation
                st.markdown(f"🔄 **Étape {i}/{len(PHASES_ORDER)}:** {phase_label}")
                st.progress(progress.percentage / 100.0)

                # Progress info
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(
                        f"📊 {progress.items_processed:,} / {progress.items_total:,} ({progress.percentage:.0f}%)"
                    )
                with col2:
                    if progress.valid_count > 0:
                        st.caption(f"💡 {progress.valid_count} stratégies valides")
            else:
                # Other phases - show spinner-like indicator
                st.markdown(f"🔄 **Étape {i}/{len(PHASES_ORDER)}:** {phase_label}")
                if progress.message:
                    st.caption(progress.message)

        else:
            # Pending phase
            st.markdown(
                f"<span style='color: #888'>⏳ Étape {i}/{len(PHASES_ORDER)}: {phase_label}</span>",
                unsafe_allow_html=True,
            )


def render_search_info_zone(stats: SearchStats | None, expanded: bool = False) -> None:
    """
    Render the persistent search info zone after search completes.

    Shows search statistics in a collapsible expander.

    Args:
        stats: Search statistics (None if no search has been run)
        expanded: Whether to expand the zone by default
    """
    if stats is None:
        return

    with st.expander("📊 Dernière recherche", expanded=expanded):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("⏱️ Durée", f"{stats.duration_seconds:.1f}s")

        with col2:
            st.metric("🧱 Briques", f"{stats.bricks_count}")

        with col3:
            st.metric("🔢 Combinaisons", f"{stats.combos_evaluated:,}")

        with col4:
            st.metric("✅ Stratégies", f"{stats.strategies_after_dedupe}")

        # Additional details
        st.caption(
            f"Mode: **{stats.mode}** | "
            f"Max biens: **{stats.max_properties}** | "
            f"Heure: **{stats.timestamp}** | "
            f"Valides: {stats.valid_strategies} → Après dédupe: {stats.strategies_after_dedupe}"
        )
