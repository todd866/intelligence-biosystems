import {
          Activity, BookOpen, BrainCircuit, ClipboardList, Database, FileDown, Gauge, GitCompareArrows, Library, Milestone, Network, Radar, Route, Sparkles, Waves
        } from "lucide-react";

        const icons = {
          Activity, BookOpen, BrainCircuit, ClipboardList, Database, FileDown, Gauge, GitCompareArrows, Library, Milestone, Network, Radar, Route, Sparkles, Waves
        };

        export const paper = {
          title: "Intelligence as High-Dimensional Coherence",
          subtitle: "The observable dimensionality bound and computational tractability",
          journal: "BioSystems",
          manuscriptId: "doi:10.1016/j.biosystems.2026.105704",
          decision: "Published",
          dueDate: "2026",
          doi: "10.1016/j.biosystems.2026.105704",
          abstract: "Biological intelligence is framed as high-dimensional coherent dynamics that remain functionally opaque to low-bandwidth external control.",
          shortThesis: "Biological intelligence is framed as high-dimensional coherent dynamics that remain functionally opaque to low-bandwidth external control.",
          mark: "I",
          pdfLabel: "PDF",
        };

        export const navLinks = [
          { href: "/", label: "Manuscript", icon: BookOpen },
          { href: "/dashboard", label: "Overview", icon: Radar },
          { href: "/revision", label: "Map", icon: ClipboardList },
          { href: "/audit", label: "Audit", icon: Gauge },
          { href: "/references", label: "References", icon: Library },
        ];

        export const downloads = [
  {
    "href": "/paper.pdf",
    "label": "PDF",
    "icon": "FileDown"
  },
  {
    "href": "/paper-source.tex",
    "label": "Source TeX",
    "icon": "FileDown"
  },
  {
    "href": "/citation-use-audit.md",
    "label": "Citation audit",
    "icon": "Database"
  },
  {
    "href": "/references.bib",
    "label": "Bibliography",
    "icon": "Library"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const headlineMetrics = [
  {
    "label": "Publication",
    "value": "BioSystems 260",
    "detail": "Article 105704 (2026)",
    "tone": "primary"
  },
  {
    "label": "Paper DOI",
    "value": "105704",
    "detail": "10.1016/j.biosystems.2026.105704",
    "tone": "tertiary"
  },
  {
    "label": "Cited sources",
    "value": "58",
    "detail": "75 bibliography entries parsed from TeX",
    "tone": "secondary"
  },
  {
    "label": "PDF/text coverage",
    "value": "58/75",
    "detail": "7 DOI-backed entries still need PaperLibrary harvest",
    "tone": "tertiary"
  }
];
        export const stanceCards = [
  {
    "title": "Observable dimensionality bound",
    "body": "External observers cannot track systems whose effective dimensionality outruns the relevant commitment channel.",
    "icon": "Radar"
  },
  {
    "title": "Sparse commitments",
    "body": "Biological systems concentrate durable record-writing into sparse boundary events rather than clocking every internal transition.",
    "icon": "Activity"
  },
  {
    "title": "Collision-free computation",
    "body": "High-dimensional dynamics avoid combinatorial collision regimes until outputs are forced through lower-dimensional channels.",
    "icon": "Network"
  },
  {
    "title": "Codes from collapse",
    "body": "Discrete biological codes emerge as reusable shadows of high-dimensional constraint dynamics.",
    "icon": "Sparkles"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const revisionPriorities = [
  {
    "title": "Make opacity functional",
    "reviewer": "Paper architecture",
    "priority": "Critical",
    "body": "Untrackability becomes a condition for autonomy rather than a mere observer inconvenience."
  },
  {
    "title": "Separate metabolic cost layers",
    "reviewer": "Paper architecture",
    "priority": "High",
    "body": "The paper distinguishes Landauer floors, real switching/maintenance costs, and commitment frequency."
  },
  {
    "title": "Use biological examples",
    "reviewer": "Paper architecture",
    "priority": "High",
    "body": "Bacteria, cortex, motor control, and code biology instantiate the same dimensional constraint."
  },
  {
    "title": "Connect to AI scaling",
    "reviewer": "Paper architecture",
    "priority": "Medium",
    "body": "Digital systems approximate high-dimensional coherence by brute-force parameterization and frequent registration."
  }
];
        export const stressTests = [
  {
    "title": "Observable dimensionality bound",
    "body": "External observers cannot track systems whose effective dimensionality outruns the relevant commitment channel.",
    "icon": "Radar"
  },
  {
    "title": "Sparse commitments",
    "body": "Biological systems concentrate durable record-writing into sparse boundary events rather than clocking every internal transition.",
    "icon": "Activity"
  },
  {
    "title": "Collision-free computation",
    "body": "High-dimensional dynamics avoid combinatorial collision regimes until outputs are forced through lower-dimensional channels.",
    "icon": "Network"
  },
  {
    "title": "Codes from collapse",
    "body": "Discrete biological codes emerge as reusable shadows of high-dimensional constraint dynamics.",
    "icon": "Sparkles"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const empiricalTests = [
  {
    "title": "Commitment bandwidth",
    "body": "How much information can the behavioral or motor channel stabilize per unit time?"
  },
  {
    "title": "Effective dimensionality",
    "body": "How many semi-independent degrees of freedom remain active in the internal dynamics?"
  },
  {
    "title": "Code formation",
    "body": "Do repeated high-dimensional encounters collapse into reusable low-dimensional symbols?"
  }
];
        export const evidenceClusters = [
  {
    "title": "Thermodynamic foundations",
    "keys": [
      "landauer1961",
      "bennett1973",
      "bennett1982",
      "todd2025maxwell"
    ],
    "icon": "Gauge"
  },
  {
    "title": "Dimensional control",
    "keys": [
      "ashby1956",
      "shannon1948",
      "cover2006",
      "gao2017"
    ],
    "icon": "Radar"
  },
  {
    "title": "Neural and biological substrates",
    "keys": [
      "miller2024",
      "pinotsis2023ephaptic",
      "levin2021",
      "attwell2001"
    ],
    "icon": "BrainCircuit"
  },
  {
    "title": "Complexity and tractability",
    "keys": [
      "bellman1961",
      "czerwinski2021",
      "bengio2013",
      "kaplan2020"
    ],
    "icon": "Network"
  }
].map((item) => ({ ...item, icon: icons[item.icon as keyof typeof icons] }));
        export const reviewerPosture = [
  {
    "reviewer": "Published article",
    "stance": "BioSystems",
    "summary": "This native site renders the full published manuscript for Intelligence as Coherence as web text with live citations and local PDF exports."
  },
  {
    "reviewer": "Reference layer",
    "stance": "PaperLibrary-backed",
    "summary": "Each cited key receives a reference page with manuscript contexts, DOI links where available, and the current local PDF/text harvest state."
  }
];
        export const reframes = [
  {
    "before": "Intelligence is output complexity.",
    "after": "Intelligence is the maintenance of coherent high-dimensional dynamics behind sparse outputs."
  },
  {
    "before": "Opacity is a limitation.",
    "after": "Opacity can protect autonomy by limiting external state control."
  },
  {
    "before": "Codes are prerequisites.",
    "after": "Codes can emerge from repeated dimensional collapse."
  }
];
        export const detailHighlights = [
  {
    "id": "main-thesis",
    "phrase": "Biological intelligence is framed as high-dimensional coherent dynamics that remain functionally opaque to low-bandwidth external control.",
    "title": "Main thesis",
    "summary": "Biological intelligence is framed as high-dimensional coherent dynamics that remain functionally opaque to low-bandwidth external control.",
    "bullets": [
      "External observers cannot track systems whose effective dimensionality outruns the relevant commitment channel.",
      "Biological systems concentrate durable record-writing into sparse boundary events rather than clocking every internal transition.",
      "High-dimensional dynamics avoid combinatorial collision regimes until outputs are forced through lower-dimensional channels."
    ]
  }
];
