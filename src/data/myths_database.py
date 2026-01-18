"""
Veritas - Mythen-Datenbank v2.0

Erweiterte Sammlung historischer Mythen und Irrtümer.
Kategorisiert nach Typ, Epoche und Region.

Enthält jetzt 25+ Mythen in verschiedenen Kategorien.
"""

from src.models.veritas_schema import (
    HistoricalMyth,
    MythOrigin,
    MythCategory,
    FactStatus,
    HistoricalEra,
    Region,
    Source,
    SourceType,
    ConfidenceLevel,
    NarrativePattern,
    MythDatabase,
)


# =============================================================================
# PERSONEN-MYTHEN
# =============================================================================

MYTH_NAPOLEON_HEIGHT = HistoricalMyth(
    id="napoleon_height",
    claim="Napoleon war klein",
    claim_en="Napoleon was short",
    category=MythCategory.PERSON_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.EUROPE],
    status=FactStatus.MYTH,
    truth="Napoleon war ca. 1,69m gross - ueberdurchschnittlich fuer franzoesische Maenner seiner Zeit (Durchschnitt: 1,64m).",
    truth_en="Napoleon was about 5'7\" (1.69m) - above average for French men of his time (average: 5'4\" / 1.64m).",
    origin=MythOrigin(
        source="Britische Propaganda und Karikaturen",
        date="1803-1815",
        reason="Kriegspropaganda zur Verspottung des Feindes",
        original_context="Der Karikaturist James Gillray stellte Napoleon als winzig dar",
        spread_mechanism="Karikaturen, Pamphlete, spaeter Populaerkultur",
    ),
    sources=[
        Source(
            type=SourceType.PRIMARY,
            title="Autopsiebericht von Dr. Francesco Antommarchi",
            year=1821,
            quote="5 Fuss 2 Zoll (franzoesisches Mass) = 1,69m",
        ),
        Source(
            type=SourceType.AUTHORITY,
            title="GND: Napoleon I., Frankreich, Kaiser",
            url="https://d-nb.info/gnd/118586408",
            reliability=ConfidenceLevel.HIGH,
        ),
        Source(
            type=SourceType.ACADEMIC,
            title="Napoleon: A Life",
            author="Andrew Roberts",
            year=2014,
        ),
    ],
    debunked_by=["Andrew Roberts", "Frank McLynn"],
    related_myths=["napoleon_complex"],
    keywords=["napoleon", "klein", "groesse", "short", "height", "bonaparte", "size"],
    popularity=85,
)

MYTH_EINSTEIN_BAD_STUDENT = HistoricalMyth(
    id="einstein_bad_student",
    claim="Einstein war ein schlechter Schueler und fiel in Mathematik durch",
    claim_en="Einstein was a bad student who failed math",
    category=MythCategory.PERSON_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.EUROPE],
    status=FactStatus.FALSE,
    truth="Einstein war ein ausgezeichneter Schueler, besonders in Mathematik und Physik. Er erhielt Bestnoten. Der Mythos entstand durch Verwechslung des Schweizer Notensystems (6=beste Note).",
    truth_en="Einstein was an excellent student, especially in math and physics. He received top grades. The myth arose from confusion about the Swiss grading system (6=best grade).",
    origin=MythOrigin(
        source="Missverstaendnis des Schweizer Notensystems",
        date="ca. 1935",
        reason="Verwechslung: In der Schweiz ist 6 die beste Note, in Deutschland die schlechteste",
        spread_mechanism="Anekdoten, Motivationsreden, Populaerkultur",
    ),
    sources=[
        Source(
            type=SourceType.PRIMARY,
            title="Einsteins Maturitaetszeugnis 1896",
            archive_id="ETH Zuerich, Hs 421:28",
            url="https://www.e-manuscripta.ch/zuznull/content/titleinfo/1481549",
            reliability=ConfidenceLevel.HIGH,
        ),
        Source(
            type=SourceType.AUTHORITY,
            title="GND: Einstein, Albert",
            url="https://d-nb.info/gnd/118529579",
            reliability=ConfidenceLevel.HIGH,
        ),
        Source(
            type=SourceType.ACADEMIC,
            title="Einstein: His Life and Universe",
            author="Walter Isaacson",
            year=2007,
        ),
    ],
    debunked_by=["Walter Isaacson", "ETH Zuerich Archive"],
    keywords=[
        "einstein",
        "schule",
        "mathematik",
        "durchgefallen",
        "failed",
        "math",
        "student",
        "genius",
    ],
    popularity=90,
)

MYTH_MARIE_ANTOINETTE_CAKE = HistoricalMyth(
    id="marie_antoinette_cake",
    claim="Marie Antoinette sagte 'Sollen sie doch Kuchen essen'",
    claim_en="Marie Antoinette said 'Let them eat cake'",
    category=MythCategory.QUOTE_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.EUROPE],
    status=FactStatus.FALSE,
    truth="Es gibt keinen Beleg, dass Marie Antoinette dies je sagte. Der Ausspruch erscheint erstmals in Rousseaus 'Bekenntnissen' (geschrieben ca. 1765, Marie Antoinette war da 9 Jahre alt) und wird einer anonymen 'grossen Prinzessin' zugeschrieben.",
    truth_en="There is no evidence Marie Antoinette ever said this. The phrase first appears in Rousseau's 'Confessions' (written c. 1765, when Marie Antoinette was 9) attributed to an anonymous 'great princess'.",
    origin=MythOrigin(
        source="Rousseaus 'Bekenntnisse'",
        date="ca. 1765/1782",
        reason="Spaetere Zuschreibung zur Daemonisierung der Koenigin",
        spread_mechanism="Revolutionaere Propaganda, spaeter Geschichtsbuecher",
    ),
    sources=[
        Source(
            type=SourceType.PRIMARY,
            title="Les Confessions",
            author="Jean-Jacques Rousseau",
            year=1782,
        ),
        Source(
            type=SourceType.ACADEMIC,
            title="Marie Antoinette: The Journey",
            author="Antonia Fraser",
            year=2001,
        ),
    ],
    debunked_by=["Antonia Fraser"],
    keywords=[
        "marie antoinette",
        "kuchen",
        "cake",
        "brioche",
        "revolution",
        "france",
        "queen",
    ],
    popularity=95,
)

MYTH_WASHINGTON_CHERRY_TREE = HistoricalMyth(
    id="washington_cherry_tree",
    claim="George Washington gestand, einen Kirschbaum gefaellt zu haben mit 'I cannot tell a lie'",
    claim_en="George Washington confessed to cutting down a cherry tree saying 'I cannot tell a lie'",
    category=MythCategory.PERSON_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.NORTH_AMERICA],
    status=FactStatus.FALSE,
    truth="Diese Geschichte wurde von Mason Locke Weems in seiner Washington-Biographie 1806 erfunden, um moralische Tugenden zu illustrieren. Es gibt keine historischen Belege fuer dieses Ereignis.",
    truth_en="This story was invented by Mason Locke Weems in his 1806 Washington biography to illustrate moral virtues. There is no historical evidence for this event.",
    origin=MythOrigin(
        source="Mason Locke Weems: 'The Life of Washington'",
        date="1806",
        reason="Moralische Erbauungsliteratur, Heldenverehrung",
        spread_mechanism="Schulbuecher, Amerikanische Folklore",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="Inventing George Washington",
            author="Edward Lengel",
            year=2011,
        ),
        Source(
            type=SourceType.FACTCHECK,
            title="Mount Vernon Website",
            url="https://www.mountvernon.org",
        ),
    ],
    debunked_by=["Edward Lengel", "Mount Vernon Estate"],
    keywords=["washington", "cherry tree", "kirschbaum", "cannot tell a lie", "honest"],
    popularity=80,
)

MYTH_NEWTON_APPLE = HistoricalMyth(
    id="newton_apple",
    claim="Newton entdeckte die Schwerkraft, als ihm ein Apfel auf den Kopf fiel",
    claim_en="Newton discovered gravity when an apple fell on his head",
    category=MythCategory.PERSON_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.EUROPE],
    status=FactStatus.MYTH,
    truth="Newton beobachtete moeglicherweise einen fallenden Apfel, aber er fiel ihm nicht auf den Kopf. Die Geschichte wurde von Voltaire popularisiert. Newton selbst erwaehnte nur, dass der Anblick eines fallenden Apfels ihn zum Nachdenken brachte.",
    truth_en="Newton may have observed a falling apple, but it didn't hit his head. The story was popularized by Voltaire. Newton himself only mentioned that seeing a falling apple made him think.",
    origin=MythOrigin(
        source="Voltaire und spaetere Biographen",
        date="ca. 1726",
        reason="Vereinfachung und Dramatisierung wissenschaftlicher Entdeckung",
        spread_mechanism="Voltaires Schriften, Biographien, Schulbuecher",
    ),
    sources=[
        Source(type=SourceType.PRIMARY, title="William Stukeleys Memoir", year=1752),
        Source(
            type=SourceType.ACADEMIC,
            title="Never at Rest: A Biography of Isaac Newton",
            author="Richard Westfall",
            year=1980,
        ),
    ],
    debunked_by=["Richard Westfall"],
    keywords=["newton", "apfel", "apple", "gravity", "schwerkraft", "head", "kopf"],
    popularity=88,
)

MYTH_CLEOPATRA_BEAUTY = HistoricalMyth(
    id="cleopatra_beauty",
    claim="Kleopatra war eine aussergewoehnliche Schoenheit",
    claim_en="Cleopatra was an exceptional beauty",
    category=MythCategory.PERSON_MYTH,
    era=HistoricalEra.ANCIENT,
    regions=[Region.MIDDLE_EAST, Region.EUROPE],
    status=FactStatus.MYTH,
    truth="Antike Quellen beschreiben Kleopatra als charismatisch und intelligent, aber nicht als aussergewoehnlich schoen. Muenzen zeigen eine Frau mit prominenter Nase. Ihre Macht lag in Intellekt, Bildung und politischem Geschick.",
    truth_en="Ancient sources describe Cleopatra as charismatic and intelligent, but not exceptionally beautiful. Coins show a woman with a prominent nose. Her power lay in intellect, education, and political skill.",
    origin=MythOrigin(
        source="Hollywood-Filme und romantisierte Darstellungen",
        date="20. Jahrhundert",
        reason="Romantisierung historischer Figuren, Orientalismus",
        spread_mechanism="Filme, Romane, Populaerkultur",
    ),
    sources=[
        Source(
            type=SourceType.PRIMARY,
            title="Plutarch: Leben des Antonius",
            author="Plutarch",
        ),
        Source(
            type=SourceType.ACADEMIC,
            title="Cleopatra: A Life",
            author="Stacy Schiff",
            year=2010,
        ),
    ],
    debunked_by=["Stacy Schiff", "Duane Roller"],
    keywords=[
        "cleopatra",
        "kleopatra",
        "beauty",
        "schoenheit",
        "egypt",
        "aegypten",
        "beautiful",
    ],
    popularity=75,
)

MYTH_VIKING_HORNS = HistoricalMyth(
    id="viking_horns",
    claim="Wikinger trugen gehoernte Helme",
    claim_en="Vikings wore horned helmets",
    category=MythCategory.PERSON_MYTH,
    era=HistoricalEra.MEDIEVAL,
    regions=[Region.EUROPE],
    status=FactStatus.FALSE,
    truth="Es gibt keine archaeologischen Beweise fuer gehoernte Wikingerhelme im Kampf. Erhaltene Helme sind hornlos. Der Mythos stammt aus dem 19. Jahrhundert, besonders aus Wagners Opern und romantischer Kunst.",
    truth_en="There is no archaeological evidence for horned Viking helmets in combat. Preserved helmets are hornless. The myth comes from the 19th century, especially Wagner's operas and romantic art.",
    origin=MythOrigin(
        source="Romantik des 19. Jahrhunderts, Wagners Opern",
        date="ca. 1870er",
        reason="Theatralische Darstellung, Nationalromantik",
        spread_mechanism="Opern, Gemaelde, spaeter Filme und Comics",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="The Viking World",
            author="Stefan Brink",
            year=2008,
        ),
        Source(
            type=SourceType.PRIMARY,
            title="Gjermundbu Helmet",
            archive_id="Museum of Cultural History, Oslo",
        ),
    ],
    debunked_by=["Roberta Frank", "Diverse Archaeologen"],
    keywords=["wikinger", "viking", "horns", "hoerner", "helmet", "helm", "horned"],
    popularity=82,
)

MYTH_EDISON_LIGHTBULB = HistoricalMyth(
    id="edison_lightbulb",
    claim="Thomas Edison erfand die Gluehbirne",
    claim_en="Thomas Edison invented the light bulb",
    category=MythCategory.PERSON_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.NORTH_AMERICA, Region.EUROPE],
    status=FactStatus.MYTH,
    truth="Edison verbesserte existierende Designs und machte die Gluehbirne praktisch nutzbar, aber er erfand sie nicht. Humphry Davy, Warren de la Rue, Joseph Swan und andere arbeiteten vorher an elektrischem Licht.",
    truth_en="Edison improved existing designs and made the light bulb practical, but didn't invent it. Humphry Davy, Warren de la Rue, Joseph Swan and others worked on electric light before him.",
    origin=MythOrigin(
        source="Amerikanische Geschichtsschreibung und Edison-PR",
        date="1880er",
        reason="Heldenverehrung, Vereinfachung komplexer Innovationsgeschichte",
        spread_mechanism="Schulbuecher, Firmengeschichte, Populaerkultur",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="The Age of Edison",
            author="Ernest Freeberg",
            year=2013,
        ),
        Source(
            type=SourceType.ACADEMIC,
            title="Empires of Light",
            author="Jill Jonnes",
            year=2003,
        ),
    ],
    debunked_by=["Diverse Technikhistoriker"],
    keywords=[
        "edison",
        "gluehbirne",
        "lightbulb",
        "light bulb",
        "invention",
        "erfindung",
        "lamp",
    ],
    popularity=78,
)

MYTH_GALILEO_PRISON = HistoricalMyth(
    id="galileo_prison",
    claim="Galileo wurde von der Kirche eingekerkert und gefoltert",
    claim_en="Galileo was imprisoned and tortured by the Church",
    category=MythCategory.PERSON_MYTH,
    era=HistoricalEra.EARLY_MODERN,
    regions=[Region.EUROPE],
    status=FactStatus.MYTH,
    truth="Galileo wurde nie gefoltert und verbrachte keinen Tag im Kerker. Nach seinem Prozess 1633 stand er unter komfortablem Hausarrest in seiner Villa bei Florenz, wo er weiter forschen und Besucher empfangen konnte.",
    truth_en="Galileo was never tortured and didn't spend a day in prison. After his 1633 trial, he was under comfortable house arrest at his villa near Florence, where he could continue research and receive visitors.",
    origin=MythOrigin(
        source="Aufklaerung und antikirchliche Polemik",
        date="18.-19. Jahrhundert",
        reason="Kirche vs. Wissenschaft Narrativ",
        spread_mechanism="Aufklaererische Schriften, spaeter Populaerkultur",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="Galileo: Watcher of the Skies",
            author="David Wootton",
            year=2010,
        ),
        Source(
            type=SourceType.ACADEMIC,
            title="The Crime of Galileo",
            author="Giorgio de Santillana",
            year=1955,
        ),
    ],
    debunked_by=["David Wootton", "John Heilbron"],
    keywords=[
        "galileo",
        "prison",
        "gefaengnis",
        "torture",
        "folter",
        "church",
        "kirche",
        "inquisition",
    ],
    popularity=70,
)

MYTH_JULIUS_CAESAR_SECTION = HistoricalMyth(
    id="caesar_birth",
    claim="Julius Caesar wurde per Kaiserschnitt geboren, daher der Name",
    claim_en="Julius Caesar was born by Caesarean section, hence the name",
    category=MythCategory.PERSON_MYTH,
    era=HistoricalEra.ANCIENT,
    regions=[Region.EUROPE],
    status=FactStatus.FALSE,
    truth="Caesars Mutter Aurelia lebte noch Jahrzehnte nach seiner Geburt - damals ueberlebten Muetter Kaiserschnitte nicht. Der Begriff 'Kaiserschnitt' kommt vermutlich vom lateinischen 'caedere' (schneiden), nicht von Caesar.",
    truth_en="Caesar's mother Aurelia lived decades after his birth - mothers didn't survive C-sections then. The term likely comes from Latin 'caedere' (to cut), not from Caesar.",
    origin=MythOrigin(
        source="Mittelalterliche Etymologie und Plinius",
        date="Mittelalter",
        reason="Falsche Etymologie und Wunsch nach dramatischer Geburtsgeschichte",
        spread_mechanism="Medizinische Texte, Etymologie-Buecher",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="A History of Medicine",
            author="Plinio Prioreschi",
            year=1996,
        ),
    ],
    debunked_by=["Medizinhistoriker"],
    keywords=["caesar", "kaiserschnitt", "caesarean", "c-section", "birth", "geburt"],
    popularity=65,
)


# =============================================================================
# KRIEGS-MYTHEN
# =============================================================================

MYTH_CLEAN_WEHRMACHT = HistoricalMyth(
    id="clean_wehrmacht",
    claim="Die Wehrmacht war sauber und nicht an NS-Verbrechen beteiligt",
    claim_en="The Wehrmacht was clean and not involved in Nazi crimes",
    category=MythCategory.WAR_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.EUROPE],
    status=FactStatus.FALSE,
    truth="Die Wehrmacht war systematisch an Kriegsverbrechen, Massenerschiessungen, der Belagerung von Leningrad und der Unterstuetzung des Holocaust beteiligt. Dies ist durch zahlreiche Dokumente, Befehle und Zeugenaussagen belegt.",
    truth_en="The Wehrmacht was systematically involved in war crimes, mass shootings, the siege of Leningrad, and supporting the Holocaust. This is documented through numerous orders, documents, and testimonies.",
    origin=MythOrigin(
        source="Nachkriegs-Apologetik und Nuernberger Verteidigungsstrategie",
        date="1945-1950er",
        reason="Rehabilitation von Veteranen, Kalter Krieg, Selbstentlastung",
        spread_mechanism="Veteranenverbaende, Memoiren, populaere Geschichtsbuecher",
    ),
    sources=[
        Source(
            type=SourceType.PRIMARY,
            title="Kommissarbefehl vom 6. Juni 1941",
            archive_id="Bundesarchiv, NS 19/3428",
            year=1941,
            reliability=ConfidenceLevel.HIGH,
        ),
        Source(
            type=SourceType.ACADEMIC,
            title="Vernichtungskrieg: Verbrechen der Wehrmacht 1941-1944",
            author="Hannes Heer",
            year=1995,
        ),
    ],
    debunked_by=["Hamburger Institut fuer Sozialforschung", "Hannes Heer"],
    related_myths=["clean_army_narrative"],
    keywords=[
        "wehrmacht",
        "clean",
        "sauber",
        "ns",
        "nazi",
        "war crimes",
        "kriegsverbrechen",
    ],
    popularity=70,
)

MYTH_DRESDEN_500K = HistoricalMyth(
    id="dresden_500k",
    claim="Bei der Bombardierung Dresdens 1945 starben 500.000 Menschen",
    claim_en="500,000 people died in the bombing of Dresden in 1945",
    category=MythCategory.WAR_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.EUROPE],
    status=FactStatus.FALSE,
    truth="Die historisch belegte Opferzahl liegt bei 22.700-25.000. Die uebertriebenen Zahlen stammen aus NS-Propaganda und wurden spaeter von Revisionisten weiterverbreitet.",
    truth_en="The historically documented death toll is 22,700-25,000. The inflated numbers came from Nazi propaganda and were later spread by revisionists.",
    origin=MythOrigin(
        source="NS-Propaganda, spaeter David Irving",
        date="1945/1960er",
        reason="Relativierung deutscher Kriegsschuld, Revisionismus",
        spread_mechanism="Propaganda, revisionistische Buecher",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="Dresden Historikerkommission Abschlussbericht",
            year=2010,
            url="https://www.dresden.de/de/leben/stadtportrait/stadtgeschichte/dresden-geschichte/historikerkommission.php",
            reliability=ConfidenceLevel.HIGH,
        ),
    ],
    debunked_by=["Dresdner Historikerkommission", "Richard Evans"],
    keywords=[
        "dresden",
        "bombing",
        "bombardierung",
        "500000",
        "opferzahl",
        "death toll",
        "ww2",
    ],
    popularity=60,
)

MYTH_FRENCH_SURRENDER = HistoricalMyth(
    id="french_surrender",
    claim="Franzosen ergeben sich immer / sind feige Soldaten",
    claim_en="French always surrender / are cowardly soldiers",
    category=MythCategory.WAR_MYTH,
    era=HistoricalEra.CONTEMPORARY,
    regions=[Region.EUROPE, Region.NORTH_AMERICA],
    status=FactStatus.FALSE,
    truth="Frankreich hat eine der erfolgreichsten Militaergeschichten Europas. Die Niederlage 1940 war ein taktisches Versagen, kein Mangel an Mut. Im Ersten Weltkrieg verlor Frankreich 1,4 Millionen Soldaten.",
    truth_en="France has one of Europe's most successful military histories. The 1940 defeat was a tactical failure, not lack of courage. In WWI, France lost 1.4 million soldiers defending their country.",
    origin=MythOrigin(
        source="Amerikanische Reaktion auf Frankreichs Irak-Kritik 2003",
        date="2003",
        reason="Politischer Streit ueber Irakkrieg, Stereotypen",
        spread_mechanism="Medien, Internet-Memes, Populaerkultur",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="A History of Modern France",
            author="Jeremy Popkin",
            year=2012,
        ),
    ],
    debunked_by=["Militaerhistoriker"],
    keywords=[
        "france",
        "frankreich",
        "surrender",
        "kapitulation",
        "feige",
        "coward",
        "white flag",
    ],
    popularity=72,
)

MYTH_SPARTANS_300 = HistoricalMyth(
    id="spartans_300",
    claim="Nur 300 Spartaner verteidigten die Thermopylen gegen Persien",
    claim_en="Only 300 Spartans defended Thermopylae against Persia",
    category=MythCategory.WAR_MYTH,
    era=HistoricalEra.ANCIENT,
    regions=[Region.EUROPE, Region.MIDDLE_EAST],
    status=FactStatus.MYTH,
    truth="Es waren etwa 7.000 Griechen am Anfang, darunter 300 Spartaner plus deren Heloten. Am letzten Tag blieben ca. 1.500 Kaempfer (300 Spartaner, 700 Thespianer, 400 Thebaner). Die '300' ignoriert die anderen griechischen Verbuendeten.",
    truth_en="There were about 7,000 Greeks initially, including 300 Spartans plus their helots. On the last day, about 1,500 fighters remained (300 Spartans, 700 Thespians, 400 Thebans). The '300' ignores other Greek allies.",
    origin=MythOrigin(
        source="Herodot (teilweise), spaeter Frank Miller's Comic",
        date="Antike / 1998",
        reason="Heldenverehrung, dramatische Vereinfachung",
        spread_mechanism="Antike Quellen, Comics, Film '300' (2006)",
    ),
    sources=[
        Source(type=SourceType.PRIMARY, title="Historien", author="Herodot", year=-440),
        Source(
            type=SourceType.ACADEMIC,
            title="Thermopylae: The Battle That Changed the World",
            author="Paul Cartledge",
            year=2006,
        ),
    ],
    debunked_by=["Paul Cartledge"],
    keywords=[
        "sparta",
        "spartaner",
        "300",
        "thermopylae",
        "thermopylen",
        "persien",
        "persia",
        "leonidas",
    ],
    popularity=85,
)

MYTH_HUMAN_TRAFFICKING_ORGANS = HistoricalMyth(
    id="ww2_soap",
    claim="Die Nazis stellten systematisch Seife aus menschlichen Koerpern her",
    claim_en="Nazis systematically made soap from human bodies",
    category=MythCategory.WAR_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.EUROPE],
    status=FactStatus.MYTH,
    truth="Es gab experimentelle Versuche in kleinem Massstab, aber keine systematische Produktion. Die Legende wurde von NS-Propaganda selbst gestreut und spaeter faelschlich als Fakt verbreitet. Holocaust-Gedenkstaetten haben dies korrigiert.",
    truth_en="There were small-scale experimental attempts, but no systematic production. The legend was spread by Nazi propaganda itself and later falsely reported as fact. Holocaust memorials have corrected this.",
    origin=MythOrigin(
        source="Geruechte und NS-Propaganda",
        date="1940er",
        reason="Geruechtebildung, spaeter unkritische Uebernahme",
        spread_mechanism="Muendliche Ueberlieferung, Medien",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="Yad Vashem FAQ",
            url="https://www.yadvashem.org",
        ),
    ],
    debunked_by=["Yad Vashem", "Joachim Neander"],
    keywords=["nazi", "soap", "seife", "holocaust", "myth"],
    popularity=50,
)


# =============================================================================
# EREIGNIS-MYTHEN
# =============================================================================

MYTH_COLUMBUS_FLAT_EARTH = HistoricalMyth(
    id="columbus_flat_earth",
    claim="Kolumbus wollte beweisen, dass die Erde rund ist",
    claim_en="Columbus wanted to prove the Earth was round",
    category=MythCategory.EVENT_MYTH,
    era=HistoricalEra.EARLY_MODERN,
    regions=[Region.EUROPE],
    status=FactStatus.FALSE,
    truth="Im 15. Jahrhundert war die Kugelgestalt der Erde unter Gebildeten allgemein bekannt (seit der Antike). Der Streit ging um den UMFANG der Erde. Kolumbus unterschaetzte ihn massiv.",
    truth_en="In the 15th century, the spherical shape of the Earth was common knowledge among educated people (known since antiquity). The dispute was about the Earth's CIRCUMFERENCE. Columbus massively underestimated it.",
    origin=MythOrigin(
        source="Washington Irvings Roman 'History of the Life and Voyages of Christopher Columbus'",
        date="1828",
        reason="Literarische Dramatisierung, Heldenverehrung",
        spread_mechanism="Schulbuecher, Populaerkultur",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="Inventing the Flat Earth",
            author="Jeffrey Burton Russell",
            year=1991,
        ),
    ],
    debunked_by=["Jeffrey Burton Russell"],
    keywords=[
        "kolumbus",
        "columbus",
        "flache erde",
        "flat earth",
        "kugelgestalt",
        "1492",
        "round",
    ],
    popularity=85,
)

MYTH_GREAT_WALL_SPACE = HistoricalMyth(
    id="great_wall_from_space",
    claim="Die Chinesische Mauer ist das einzige menschliche Bauwerk, das man vom Weltraum sieht",
    claim_en="The Great Wall of China is the only man-made structure visible from space",
    category=MythCategory.EVENT_MYTH,
    era=HistoricalEra.CONTEMPORARY,
    regions=[Region.ASIA],
    status=FactStatus.FALSE,
    truth="Die Mauer ist vom Weltraum aus ohne Hilfsmittel NICHT sichtbar (zu schmal, ca. 5-8m). Viele andere Strukturen (Strassen, Staedte) sind hingegen sichtbar. Mehrere Astronauten haben dies bestaetigt.",
    truth_en="The Wall is NOT visible from space without aid (too narrow, about 5-8m). Many other structures (roads, cities) are visible. Multiple astronauts have confirmed this.",
    origin=MythOrigin(
        source="Diverse fruehe Quellen, u.a. Henry Norman (1895)",
        date="vor dem Weltraumzeitalter",
        reason="Mythenbildung ueber ein beeindruckendes Bauwerk",
        spread_mechanism="Schulbuecher, Tourismus-Werbung",
    ),
    sources=[
        Source(
            type=SourceType.PRIMARY,
            title="Aussage von Yang Liwei (erster chinesischer Astronaut)",
            year=2003,
        ),
        Source(
            type=SourceType.PRIMARY, title="NASA Statement", url="https://www.nasa.gov"
        ),
    ],
    debunked_by=["Yang Liwei", "NASA"],
    keywords=[
        "chinesische mauer",
        "great wall",
        "weltraum",
        "space",
        "sichtbar",
        "visible",
        "china",
    ],
    popularity=90,
)

MYTH_TITANIC_UNSINKABLE = HistoricalMyth(
    id="titanic_unsinkable",
    claim="Die Titanic wurde als 'unsinkbar' beworben",
    claim_en="The Titanic was advertised as 'unsinkable'",
    category=MythCategory.EVENT_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.EUROPE, Region.NORTH_AMERICA],
    status=FactStatus.MYTH,
    truth="White Star Line hat die Titanic nie offiziell als 'unsinkbar' beworben. Der Begriff erschien in einigen Zeitungsartikeln, aber nicht in der offiziellen Werbung. Der Mythos verstaerkte sich nach dem Untergang.",
    truth_en="White Star Line never officially advertised the Titanic as 'unsinkable'. The term appeared in some newspaper articles, but not in official advertising. The myth strengthened after the sinking.",
    origin=MythOrigin(
        source="Zeitungsartikel und spaetere Dramatisierung",
        date="1912",
        reason="Dramatische Ironie nach dem Untergang",
        spread_mechanism="Medien, Filme, Buecher",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="A Night to Remember",
            author="Walter Lord",
            year=1955,
        ),
        Source(
            type=SourceType.ACADEMIC,
            title="Titanic: The Ship Magnificent",
            author="Bruce Beveridge",
            year=2008,
        ),
    ],
    debunked_by=["Richard Howells"],
    keywords=[
        "titanic",
        "unsinkable",
        "unsinkbar",
        "ship",
        "schiff",
        "iceberg",
        "eisberg",
    ],
    popularity=88,
)

MYTH_MEDIEVAL_FLAT_EARTH = HistoricalMyth(
    id="medieval_flat_earth",
    claim="Menschen im Mittelalter glaubten, die Erde sei flach",
    claim_en="Medieval people believed the Earth was flat",
    category=MythCategory.EVENT_MYTH,
    era=HistoricalEra.MEDIEVAL,
    regions=[Region.EUROPE],
    status=FactStatus.FALSE,
    truth="Gebildete Menschen im Mittelalter wussten, dass die Erde eine Kugel ist. Dies war seit der Antike bekannt. Der Mythos wurde im 19. Jahrhundert erfunden, um das Mittelalter als 'dunkel' darzustellen.",
    truth_en="Educated medieval people knew the Earth was a sphere. This was known since antiquity. The myth was invented in the 19th century to portray the Middle Ages as 'dark'.",
    origin=MythOrigin(
        source="Washington Irving und spaetere Autoren",
        date="19. Jahrhundert",
        reason="Fortschrittsnarrativ der Aufklaerung",
        spread_mechanism="Schulbuecher, Populaerkultur",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="Inventing the Flat Earth",
            author="Jeffrey Burton Russell",
            year=1991,
        ),
    ],
    debunked_by=["Jeffrey Burton Russell", "Diverse Mediaevisten"],
    keywords=[
        "mittelalter",
        "medieval",
        "flat earth",
        "flache erde",
        "middle ages",
        "belief",
    ],
    popularity=75,
)

MYTH_SALEM_BURNING = HistoricalMyth(
    id="salem_burning",
    claim="In Salem wurden Hexen auf dem Scheiterhaufen verbrannt",
    claim_en="Witches were burned at the stake in Salem",
    category=MythCategory.EVENT_MYTH,
    era=HistoricalEra.EARLY_MODERN,
    regions=[Region.NORTH_AMERICA],
    status=FactStatus.FALSE,
    truth="In Salem (1692) wurde niemand verbrannt. 19 Menschen wurden gehaengt, einer durch Steine zu Tode gedrueckt. Hexenverbrennungen waren primaer ein europaeisches Phaenomen.",
    truth_en="No one was burned in Salem (1692). 19 people were hanged, one was pressed to death with stones. Witch burnings were primarily a European phenomenon.",
    origin=MythOrigin(
        source="Vermischung mit europaeischen Hexenprozessen",
        date="Spaeteres 17. Jahrhundert",
        reason="Verwechslung verschiedener Traditionen",
        spread_mechanism="Populaerkultur, falsche Schulbuchdarstellungen",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="The Salem Witch Trials",
            author="Marilynne Roach",
            year=2002,
        ),
    ],
    debunked_by=["Salem Witch Museum", "Marilynne Roach"],
    keywords=[
        "salem",
        "witch",
        "hexe",
        "burning",
        "verbrennung",
        "stake",
        "scheiterhaufen",
    ],
    popularity=68,
)

MYTH_NERO_FIDDLE = HistoricalMyth(
    id="nero_fiddle",
    claim="Kaiser Nero spielte Geige, waehrend Rom brannte",
    claim_en="Emperor Nero played the fiddle while Rome burned",
    category=MythCategory.EVENT_MYTH,
    era=HistoricalEra.ANCIENT,
    regions=[Region.EUROPE],
    status=FactStatus.FALSE,
    truth="Die Geige wurde erst 1.500 Jahre spaeter erfunden. Nero war beim Brand (64 n.Chr.) nicht einmal in Rom. Er soll jedoch bei anderen Gelegenheiten Lyra gespielt und gesungen haben.",
    truth_en="The fiddle wasn't invented until 1,500 years later. Nero wasn't even in Rome during the fire (64 AD). He reportedly played the lyre and sang on other occasions.",
    origin=MythOrigin(
        source="Spaetere Umdichtung antiker Berichte",
        date="Mittelalter/Neuzeit",
        reason="Anachronismus, Vermischung von Instrumenten",
        spread_mechanism="Volksetymologie, Populaerkultur",
    ),
    sources=[
        Source(type=SourceType.PRIMARY, title="Annalen", author="Tacitus"),
        Source(
            type=SourceType.ACADEMIC, title="Nero", author="Edward Champlin", year=2003
        ),
    ],
    debunked_by=["Edward Champlin"],
    keywords=["nero", "fiddle", "geige", "rome", "rom", "fire", "brand", "burning"],
    popularity=78,
)


# =============================================================================
# URSPRUNGS-MYTHEN
# =============================================================================

MYTH_DARK_AGES = HistoricalMyth(
    id="dark_ages",
    claim="Das Mittelalter war eine Zeit des Stillstands und der Unwissenheit",
    claim_en="The Middle Ages were a time of stagnation and ignorance",
    category=MythCategory.ORIGIN_MYTH,
    era=HistoricalEra.MEDIEVAL,
    regions=[Region.EUROPE],
    status=FactStatus.MYTH,
    truth="Das Mittelalter brachte viele Innovationen: Universitaeten, gotische Architektur, Agrarrevolution, Brillen, mechanische Uhren. Der Begriff 'Dunkles Zeitalter' stammt von Renaissance-Humanisten.",
    truth_en="The Middle Ages brought many innovations: universities, Gothic architecture, agricultural revolution, eyeglasses, mechanical clocks. The term 'Dark Ages' came from Renaissance humanists.",
    origin=MythOrigin(
        source="Renaissance-Humanisten (Petrarca)",
        date="14. Jahrhundert",
        reason="Selbstdarstellung der Renaissance als 'Wiedergeburt'",
        spread_mechanism="Aufklaerung, Schulbuecher, Populaerkultur",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="The Light Ages",
            author="Seb Falk",
            year=2020,
        ),
        Source(
            type=SourceType.ACADEMIC,
            title="Medieval Technology and Social Change",
            author="Lynn White Jr.",
            year=1962,
        ),
    ],
    debunked_by=["Moderne Mediaevistik"],
    keywords=[
        "mittelalter",
        "dark ages",
        "dunkles zeitalter",
        "stagnation",
        "medieval",
        "ignorance",
    ],
    popularity=75,
)

MYTH_AMERICAN_INVENTION = HistoricalMyth(
    id="american_invention_democracy",
    claim="Amerika erfand die Demokratie",
    claim_en="America invented democracy",
    category=MythCategory.ORIGIN_MYTH,
    era=HistoricalEra.MODERN,
    regions=[Region.NORTH_AMERICA],
    status=FactStatus.FALSE,
    truth="Demokratie existierte im antiken Griechenland (Athen, 5. Jh. v.Chr.). Die USA schufen eine repraesentative Republik mit demokratischen Elementen, beeinflusst von griechischen, roemischen und britischen Vorbildern.",
    truth_en="Democracy existed in ancient Greece (Athens, 5th c. BC). The US created a representative republic with democratic elements, influenced by Greek, Roman, and British models.",
    origin=MythOrigin(
        source="Amerikanischer Exzeptionalismus",
        date="19.-20. Jahrhundert",
        reason="Nationalstolz, vereinfachte Geschichtsdarstellung",
        spread_mechanism="Schulbuecher, politische Rhetorik",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="Democracy: A History",
            author="John Dunn",
            year=2005,
        ),
    ],
    debunked_by=["Politikwissenschaftler"],
    keywords=[
        "america",
        "amerika",
        "democracy",
        "demokratie",
        "invention",
        "erfindung",
        "greece",
    ],
    popularity=55,
)

MYTH_BLOOD_RED_BLUE = HistoricalMyth(
    id="blood_blue",
    claim="Sauerstoffarmes Blut ist blau",
    claim_en="Deoxygenated blood is blue",
    category=MythCategory.ORIGIN_MYTH,
    era=HistoricalEra.CONTEMPORARY,
    regions=[Region.GLOBAL],
    status=FactStatus.FALSE,
    truth="Menschliches Blut ist immer rot. Sauerstoffarmes Blut ist dunkelrot, nicht blau. Adern erscheinen blau durch die Haut wegen Lichtabsorption und -streuung, nicht wegen der Blutfarbe.",
    truth_en="Human blood is always red. Deoxygenated blood is dark red, not blue. Veins appear blue through skin due to light absorption and scattering, not blood color.",
    origin=MythOrigin(
        source="Schulbiologie-Vereinfachungen und Diagramme",
        date="20. Jahrhundert",
        reason="Didaktische Vereinfachung in Schaubildern",
        spread_mechanism="Schulbuecher, Anatomie-Diagramme",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC, title="Medical Physiology", author="Guyton & Hall"
        ),
    ],
    debunked_by=["Mediziner"],
    keywords=[
        "blood",
        "blut",
        "blue",
        "blau",
        "red",
        "rot",
        "oxygen",
        "sauerstoff",
        "veins",
    ],
    popularity=65,
)


# =============================================================================
# TECHNOLOGIE-MYTHEN
# =============================================================================

MYTH_INTERNET_GORE = HistoricalMyth(
    id="internet_al_gore",
    claim="Al Gore behauptete, das Internet erfunden zu haben",
    claim_en="Al Gore claimed he invented the Internet",
    category=MythCategory.QUOTE_MYTH,
    era=HistoricalEra.CONTEMPORARY,
    regions=[Region.NORTH_AMERICA],
    status=FactStatus.MYTH,
    truth="Gore sagte, er habe 'die Initiative ergriffen, das Internet zu schaffen' - gemeint war seine politische Foerderung, nicht technische Erfindung. Die 'Erfinder' Cerf und Kahn verteidigten Gore.",
    truth_en="Gore said he 'took the initiative in creating the Internet' - meaning his political support, not technical invention. The actual inventors Cerf and Kahn defended Gore.",
    origin=MythOrigin(
        source="Politische Gegner, Medien-Vereinfachung",
        date="1999",
        reason="Wahlkampf-Attacke, Missverstaendnis",
        spread_mechanism="Medien, politische Satire, Internet-Memes",
    ),
    sources=[
        Source(type=SourceType.PRIMARY, title="CNN Interview Transkript", year=1999),
        Source(
            type=SourceType.ACADEMIC,
            title="Statement von Vint Cerf und Bob Kahn",
            year=2000,
        ),
    ],
    debunked_by=["Vint Cerf", "Bob Kahn", "Snopes"],
    keywords=["al gore", "internet", "invented", "erfunden", "created", "initiative"],
    popularity=70,
)

MYTH_5_SECOND_RULE = HistoricalMyth(
    id="five_second_rule",
    claim="Essen, das weniger als 5 Sekunden am Boden liegt, ist noch sauber",
    claim_en="Food dropped on the floor for less than 5 seconds is still clean",
    category=MythCategory.ORIGIN_MYTH,
    era=HistoricalEra.CONTEMPORARY,
    regions=[Region.GLOBAL],
    status=FactStatus.FALSE,
    truth="Bakterien uebertragen sich sofort bei Kontakt. Studien zeigen, dass Kontamination in Millisekunden stattfindet. Die Menge haengt von Oberflaeche und Feuchtigkeit ab, nicht von der Zeit.",
    truth_en="Bacteria transfer immediately on contact. Studies show contamination occurs in milliseconds. The amount depends on surface and moisture, not time.",
    origin=MythOrigin(
        source="Volksweisheit, moeglicherweise Dschingis Khan Legende",
        date="Unbekannt",
        reason="Wunschdenken, Bequemlichkeit",
        spread_mechanism="Muendliche Ueberlieferung",
    ),
    sources=[
        Source(
            type=SourceType.ACADEMIC,
            title="Rutgers University Study",
            author="Donald Schaffner",
            year=2016,
        ),
    ],
    debunked_by=["Donald Schaffner", "Rutgers University"],
    keywords=[
        "5 second rule",
        "5 sekunden regel",
        "floor",
        "boden",
        "bacteria",
        "bakterien",
        "food",
    ],
    popularity=80,
)


# =============================================================================
# NARRATIVE PATTERNS
# =============================================================================

NARRATIVE_CLEAN_ARMY = NarrativePattern(
    id="clean_army",
    name="Saubere Armee",
    name_en="Clean Army Myth",
    description="Das Narrativ, dass die regulaere Armee eines Staates nicht an Verbrechen beteiligt war, die nur von speziellen Einheiten begangen wurden.",
    typical_claims=[
        "Die Armee hat nur Befehle befolgt",
        "Die Verbrechen wurden von anderen begangen",
        "Die einfachen Soldaten wussten von nichts",
    ],
    keywords=[
        "saubere armee",
        "clean army",
        "nur befehle",
        "just orders",
        "wussten nichts",
    ],
    origin="Nuernberger Prozesse 1945-1946",
    purpose="Entlastung von Veteranen und Institutionen",
    examples=["clean_wehrmacht"],
    counter_narrative="Historische Forschung zeigt systematische Beteiligung regulaerer Armeen an Kriegsverbrechen.",
)

NARRATIVE_VICTIM_NUMBERS = NarrativePattern(
    id="victim_numbers_inflation",
    name="Opferzahlen-Inflation",
    name_en="Victim Numbers Inflation",
    description="Das Narrativ, Opferzahlen von Ereignissen stark zu uebertreiben, um politische Punkte zu machen.",
    typical_claims=[
        "Es waren viel mehr Tote als offiziell zugegeben",
        "Die wahren Zahlen werden verschwiegen",
        "Historiker luegen ueber die Opferzahlen",
    ],
    keywords=["opferzahlen", "victim numbers", "verschwiegen", "covered up"],
    purpose="Delegitimierung von Gegnern, Opferkonkurrenz",
    examples=["dresden_500k"],
)

NARRATIVE_GREAT_LEADER = NarrativePattern(
    id="great_leader_myth",
    name="Grosser-Fuehrer-Mythos",
    name_en="Great Leader Myth",
    description="Das Narrativ, dass historische Veraenderungen primaer auf einzelne 'grosse Maenner' zurueckzufuehren sind.",
    typical_claims=[
        "X hat im Alleingang Y erreicht",
        "Ohne X waere Y nie passiert",
        "X war ein Genie, das seiner Zeit voraus war",
    ],
    keywords=["grosser mann", "great man", "genie", "genius", "visionaer", "visionary"],
    purpose="Vereinfachung komplexer Geschichte, Heldenverehrung",
    counter_narrative="Geschichte wird von vielen Faktoren geformt: Strukturen, Gruppen, Zufaelle, nicht nur Individuen.",
)

NARRATIVE_GOLDEN_AGE = NarrativePattern(
    id="golden_age",
    name="Goldenes Zeitalter",
    name_en="Golden Age Myth",
    description="Das Narrativ, dass fruehere Zeiten besser, reiner oder moralischer waren als die Gegenwart.",
    typical_claims=[
        "Frueher war alles besser",
        "Die Menschen waren frueher ehrlicher/fleissiger",
        "Die Gesellschaft ist verfallen",
    ],
    keywords=[
        "frueher war alles besser",
        "golden age",
        "verfall",
        "decline",
        "good old days",
    ],
    purpose="Kritik an Gegenwart, Nostalgie, politische Mobilisierung",
    counter_narrative="Jede Epoche hatte ihre Probleme. Viele Metriken (Gesundheit, Lebenserwartung, Gewalt) haben sich verbessert.",
)

NARRATIVE_SUPPRESSED_TRUTH = NarrativePattern(
    id="suppressed_truth",
    name="Unterdrueckte Wahrheit",
    name_en="Suppressed Truth",
    description="Das Narrativ, dass wichtige historische Wahrheiten aktiv unterdrueckt werden.",
    typical_claims=[
        "Das wird uns nicht gesagt",
        "Die Mainstream-Historiker verschweigen...",
        "Die wahre Geschichte ist...",
    ],
    keywords=[
        "verschwiegen",
        "suppressed",
        "hidden history",
        "verborgene geschichte",
        "mainstream",
    ],
    purpose="Diskreditierung etablierter Forschung, Alternative Narrative",
    counter_narrative="Historische Forschung ist offen und selbstkorrigierend. Neue Erkenntnisse werden publiziert und debattiert.",
)


# =============================================================================
# Datenbank initialisieren
# =============================================================================


def create_initial_database() -> MythDatabase:
    """Erstellt die erweiterte Mythen-Datenbank v2.0."""

    db = MythDatabase()

    # Alle Mythen hinzufuegen
    myths = [
        # Personen-Mythen
        MYTH_NAPOLEON_HEIGHT,
        MYTH_EINSTEIN_BAD_STUDENT,
        MYTH_MARIE_ANTOINETTE_CAKE,
        MYTH_WASHINGTON_CHERRY_TREE,
        MYTH_NEWTON_APPLE,
        MYTH_CLEOPATRA_BEAUTY,
        MYTH_VIKING_HORNS,
        MYTH_EDISON_LIGHTBULB,
        MYTH_GALILEO_PRISON,
        MYTH_JULIUS_CAESAR_SECTION,
        # Kriegs-Mythen
        MYTH_CLEAN_WEHRMACHT,
        MYTH_DRESDEN_500K,
        MYTH_FRENCH_SURRENDER,
        MYTH_SPARTANS_300,
        MYTH_HUMAN_TRAFFICKING_ORGANS,
        # Ereignis-Mythen
        MYTH_COLUMBUS_FLAT_EARTH,
        MYTH_GREAT_WALL_SPACE,
        MYTH_TITANIC_UNSINKABLE,
        MYTH_MEDIEVAL_FLAT_EARTH,
        MYTH_SALEM_BURNING,
        MYTH_NERO_FIDDLE,
        # Ursprungs-Mythen
        MYTH_DARK_AGES,
        MYTH_AMERICAN_INVENTION,
        MYTH_BLOOD_RED_BLUE,
        # Technologie-Mythen
        MYTH_INTERNET_GORE,
        MYTH_5_SECOND_RULE,
    ]

    for myth in myths:
        db.myths[myth.id] = myth

    # Alle Narrative hinzufuegen
    narratives = [
        NARRATIVE_CLEAN_ARMY,
        NARRATIVE_VICTIM_NUMBERS,
        NARRATIVE_GREAT_LEADER,
        NARRATIVE_GOLDEN_AGE,
        NARRATIVE_SUPPRESSED_TRUTH,
    ]

    for narrative in narratives:
        db.narratives[narrative.id] = narrative

    return db


# Globale Datenbank-Instanz
MYTHS_DATABASE = create_initial_database()


def get_myths_database() -> MythDatabase:
    """Gibt die Mythen-Datenbank zurueck."""
    return MYTHS_DATABASE


# =============================================================================
# Statistics Helper
# =============================================================================


def get_database_stats() -> dict:
    """Gibt Statistiken ueber die Datenbank zurueck."""
    db = MYTHS_DATABASE

    stats = {
        "total_myths": len(db.myths),
        "total_narratives": len(db.narratives),
        "myths_by_category": {},
        "myths_by_era": {},
        "myths_by_status": {},
    }

    for myth in db.myths.values():
        # By category
        cat = (
            myth.category.value
            if hasattr(myth.category, "value")
            else str(myth.category)
        )
        stats["myths_by_category"][cat] = stats["myths_by_category"].get(cat, 0) + 1

        # By era
        era = myth.era.value if hasattr(myth.era, "value") else str(myth.era)
        stats["myths_by_era"][era] = stats["myths_by_era"].get(era, 0) + 1

        # By status
        status = (
            myth.status.value if hasattr(myth.status, "value") else str(myth.status)
        )
        stats["myths_by_status"][status] = stats["myths_by_status"].get(status, 0) + 1

    return stats
