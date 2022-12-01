# Project information

project = "HPLOP"
copyright = "2022, Alfonso Sánchez Rodríguez"
author = "Alfonso Sánchez Rodríguez"
release = "0.1.0a"
version = "0.1"

# General configuration

extensions = ["sphinx_design"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output configuration

html_theme = "pydata_sphinx_theme"
html_title = f"{project} Manual"
html_static_path = ["_static"]
html_css_files = ["hplop.css"]

html_theme_options = {
    "navbar_persistent": [],
    "navbar_end": ["navbar-icon-links"],
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "Github",
            "url": "https://github.com/alfonsoSR/hplop",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
}

html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs", "sidebar-ethical-ads"],
}

html_context = {
    "github_user": "alfonsoSR",
    "github_repo": "hplop",
    "github_version": "documentation",
    "doc_path": "docs",
    "default_mode": "light",
}
