"""Filter components for property and strategy filtering."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable
import streamlit as st


def get_unique_values(archetypes: List[Dict[str, Any]], key: str) -> List[str]:
    """Extract unique values for a field from archetypes.
    
    Args:
        archetypes: List of archetype dicts
        key: Field name to extract
        
    Returns:
        Sorted list of unique non-empty values
    """
    values = {a.get(key) for a in archetypes if a.get(key)}
    return sorted(str(v) for v in values)


def prettify_mode_loyer(code: str) -> str:
    """Convert mode_loyer code to human-readable label.
    
    Args:
        code: Internal code like 'meuble_classique'
        
    Returns:
        Formatted label like 'Meuble Classique'
    """
    if not code:
        return "—"
    text = code.replace("_", " ").replace("-", " ").strip()
    tokens = []
    for t in text.split():
        if t.lower() in {"lmnp", "bic", "lcd"}:
            tokens.append(t.upper())
        else:
            tokens.append(t.capitalize())
    return " ".join(tokens)


def render_property_filters(
    archetypes: List[Dict[str, Any]],
) -> Tuple[List[str], List[str], bool]:
    """Render property filter controls.
    
    Args:
        archetypes: Full list of archetypes to filter
        
    Returns:
        Tuple of (selected_villes, selected_types, apply_rent_cap)
    """
    st.markdown("### Sélection du périmètre")
    
    # Get unique values
    villes = get_unique_values(archetypes, "ville")
    types_bien = get_unique_values(archetypes, "mode_loyer")
    
    # Build label map from data
    code_to_label = {}
    for a in archetypes:
        code = str(a.get("mode_loyer", "")).strip()
        lbl = a.get("mode_loyer_label")
        if code and lbl:
            code_to_label[code] = lbl
    
    def label_for(code: str) -> str:
        return code_to_label.get(code, prettify_mode_loyer(code))
    
    # City filter
    selected_villes = st.multiselect(
        "Filtrer les villes",
        villes,
        default=villes if len(villes) <= 12 else [],
    )
    
    # Type filter
    selected_types = st.multiselect(
        "Filtrer les types de bien",
        options=types_bien,
        default=types_bien,
        format_func=label_for,
        help="Sélectionnez les familles de stratégies à considérer.",
    )
    
    # Rent cap
    apply_rent_cap = st.checkbox(
        "Appliquer l'encadrement des loyers si configuré",
        value=True,
    )
    
    return selected_villes, selected_types, apply_rent_cap


def filter_archetypes(
    archetypes: List[Dict[str, Any]],
    villes: Optional[List[str]] = None,
    types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Filter archetypes by ville and mode_loyer.
    
    Args:
        archetypes: Full list to filter
        villes: Selected villes (None = all)
        types: Selected mode_loyer types (None = all)
        
    Returns:
        Filtered list
    """
    result = archetypes
    
    if villes:
        result = [a for a in result if a.get("ville") in villes]
    
    if types:
        result = [
            a for a in result 
            if str(a.get("mode_loyer", "")).strip() in types
        ]
    
    return result
