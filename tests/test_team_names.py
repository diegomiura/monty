from src.data.team_names import ALIASES, load_alias_table, make_resolver


def test_static_aliases_resolve():
    r = make_resolver(load_alias_table(None))
    assert r("USA") == "United States"
    assert r("Côte d'Ivoire") == "Ivory Coast"
    assert r("Bosnia & Herzegovina") == "Bosnia and Herzegovina"
    assert r("Korea Republic") == "South Korea"
    assert r("Türkiye") == "Turkey"


def test_unknown_names_pass_through():
    r = make_resolver(load_alias_table(None))
    assert r("Atlantis") == "Atlantis"


def test_former_names_added(tmp_path):
    csv = tmp_path / "former.csv"
    csv.write_text("current,former,start_date,end_date\nDR Congo,Zaïre,1971-01-10,1997-04-27\n")
    r = make_resolver(load_alias_table(csv))
    assert r("Zaïre") == "DR Congo"


def test_no_alias_maps_to_alias():
    """Alias targets must themselves be canonical (no chains)."""
    for target in ALIASES.values():
        assert target not in ALIASES
