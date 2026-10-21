"""
Unified relation → category mapping for multiple datasets.

**FB15k237**: Freebase path format (`/film/film/language` → domain = `film`).
**WN18RR**:    WordNet relation format (`_hypernym` → semantic category).

Both use the same 4-target-category convention: 3 explicit + 'other'.
"""
import os

# ========== FB15k237: Freebase domain categories ==========

# Target categories for FB15k237
FB15k237_TARGET_CATEGORIES = ['film', 'people', 'common', 'other']

# All valid Freebase domain categories (non-target domains map to 'other')
CATEGORY_DOMAINS = {
    'film': 'film',
    'people': 'people',
    'location': 'location',
    'sports': 'sports',
    'education': 'education',
    'award': 'award',
    'music': 'music',
    'tv': 'tv',
    'organization': 'organization',
    'government': 'government',
    'olympics': 'olympics',
    'business': 'business',
    'base': 'base',
    'medicine': 'medicine',
    'soccer': 'soccer',
    'common': 'common',
}


def get_category_from_path(relation_path: str) -> str:
    """Extract the Freebase domain category from a relation path.

    Args:
        relation_path: e.g. '/film/film/language'

    Returns:
        Category string (e.g. 'film', 'people'), or 'other'.
    """
    if not relation_path or not relation_path.startswith('/'):
        return 'other'

    parts = relation_path.split('/')
    if len(parts) < 2:
        return 'other'

    domain = parts[1]
    return domain if domain in CATEGORY_DOMAINS else 'other'


# ========== WN18RR: WordNet semantic categories ==========

# Target categories for WN18RR (3 explicit + other)
WN18RR_TARGET_CATEGORIES = ['hierarchical', 'part_whole', 'domain', 'other']

# WN18RR relation name → category mapping
WN18RR_RELATION_CATEGORY = {
    '_hypernym':                   'hierarchical',
    '_instance_hypernym':          'hierarchical',
    '_similar_to':                 'hierarchical',
    '_has_part':                   'part_whole',
    '_member_meronym':             'part_whole',
    '_member_of_domain_usage':     'domain',
    '_member_of_domain_region':    'domain',
    '_synset_domain_topic_of':     'domain',
    '_derivationally_related_form': 'other',
    '_also_see':                   'other',
    '_verb_group':                 'other',
}


def get_wn18rr_category(relation_name: str) -> str:
    """Classify a WN18RR relation name into a semantic category.

    Args:
        relation_name: e.g. '_hypernym', '_has_part'

    Returns:
        One of: 'hierarchical', 'part_whole', 'domain', 'other'
    """
    return WN18RR_RELATION_CATEGORY.get(relation_name, 'other')


# ========== Unified dispatcher ==========

def get_category(relation_name: str, data_type: str = None) -> str:
    """Auto-detect the dataset format and classify accordingly.

    Detection logic:
        - Starts with '/'  → FB15k237 (Freebase path) → use get_category_from_path
        - Starts with '_'  → WN18RR (WordNet relation) → use get_wn18rr_category
        - Explicit data_type provided → use that dataset's classifier

    Args:
        relation_name: The relation string (e.g. '/film/film/language' or '_hypernym')
        data_type:     Optional dataset name ('FB15k237' or 'WN18RR') to force a classifier.

    Returns:
        Category string.
    """
    if data_type == 'WN18RR' or (data_type is None and relation_name.startswith('_')):
        return get_wn18rr_category(relation_name)
    return get_category_from_path(relation_name)


def get_target_categories(data_type: str = None) -> list:
    """Return the appropriate target categories list for a dataset."""
    if data_type == 'WN18RR':
        return WN18RR_TARGET_CATEGORIES
    return FB15k237_TARGET_CATEGORIES


def get_category_stats(relations_file: str = None, data_type: str = None) -> dict:
    """
    Compute per-category statistics from a relations.txt file.

    Args:
        relations_file: Path to relations.txt. If None, defaults to FB15k237 path.
        data_type: 'FB15k237' or 'WN18RR' to select the classifier.

    Returns:
        dict mapping category → {'count': int, 'examples': [str, ...]}
    """
    if relations_file is None:
        # Default to FB15k237
        relations_file = os.path.join(
            os.path.dirname(__file__),
            '..', '..', 'datasets', 'FB15k237', 'EditKnowledge', 'relations.txt'
        )

    stats = {}
    with open(relations_file) as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            cat = get_category(rel, data_type)
            if cat not in stats:
                stats[cat] = {'count': 0, 'examples': []}
            stats[cat]['count'] += 1
            if len(stats[cat]['examples']) < 3:
                stats[cat]['examples'].append(rel)

    return stats


def _print_category_stats(stats: dict, target_cats: list, title: str):
    """Pretty-print category statistics."""
    print(f"=== {title} ===\n")
    for cat in sorted(stats.keys(), key=lambda c: -stats[c]['count']):
        info = stats[cat]
        marker = ' ★' if cat in target_cats else ''
        print(f"  {cat:20s} | {info['count']:3d} relations{marker}")
        for ex in info['examples']:
            print(f"  {'':20s}   {ex}")
        print()

    total = sum(v['count'] for v in stats.values())
    print(f"  Total: {total} relations across {len(stats)} categories")
    target_total = sum(stats[c]['count'] for c in target_cats if c in stats)
    print(f"  Target categories: {target_total} relations ({target_total / total * 100:.1f}%)")


if __name__ == '__main__':
    # FB15k237
    stats_fb = get_category_stats(data_type='FB15k237')
    _print_category_stats(stats_fb, FB15k237_TARGET_CATEGORIES,
                          'FB15k237 Domain Category Distribution')

    # WN18RR
    print("\n" + "=" * 60 + "\n")
    wn18rr_file = os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'datasets', 'WN18RR', 'EditKnowledge', 'relations.txt'
    )
    if os.path.exists(wn18rr_file):
        stats_wn = get_category_stats(wn18rr_file, data_type='WN18RR')
        _print_category_stats(stats_wn, WN18RR_TARGET_CATEGORIES,
                              'WN18RR Semantic Category Distribution')
