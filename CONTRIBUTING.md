Contributing
------------

Please use the following guidelines when contributing to this project. 

Before contributing signficicant changes, please begin a discussion of the
desired changes via a GitHub Issue to prevent doing unnecessary or overlapping
work.

## License

The preferred license for source code contributed to this project is the Apache
License 2.0 (https://www.apache.org/licenses/LICENSE-2.0) and for
documentation, including Jupyter notebooks and text documentation, is the
Creative Commons Attribution 4.0 International (CC BY 4.0)
(https://creativecommons.org/licenses/by/4.0/). Contributions under other,
compatible licenses will be considered on a case-by-case basis.

Contributions must include a "signed off by" tag in the commit message for the
contributions asserting the signing of the developers certificate of origin
(https://developercertificate.org/). A GPG-signed commit with the "signed off
by" tag is preferred.

## Styling

Please use the following style guidelines when making contributions.

### Source Code

* Two-space indention, no tabs
* To the extent possible, variable names should be descriptive
* Fortran codes should use free-form source files
* Fortran codes should not use implicit variable names and should use 
  `implicit none`
* The following file extensions should be used appropriately
  * C - `.c`
  * C++ - `.cpp`
  * CUDA C/C++ - `.cu`
  * CUDA Fortran - `.cuf`
  * Fortran - `.F90`

### Jupyter Notebooks & Markdown

* When they appear inline with the text; directive names, clauses, function or
  subroutine names, variable names, file names, commands and command-line
  arguments should appear between two back ticks.
* Code blocks should begin with three back ticks and either 'cpp' or 'fortran'
  to enable appropriate source formatting and end with three back ticks.
* Emphasis, including quotes made for emphasis and introduction of new terms
  should be highlighted between a single pair of asterisks
* A level 1 heading should appear at the top of the notebook as the title of
  the notebook.
* A horizontal rule should appear between sections that begin with a level 2
  heading.
* When a lab participant needs to open a file for editing in a notebook, do not
  use jupyter's edit links, but rather use the following convention:
  `**<link text>** (File -> Open -> <path-to-file>)`. This eliminates issues
  with links breaking when jupyter is launched from a different directory. The
  `generate_markdown` script will look for this convention to fix the links in
  markdown.
* If a cell should not be included in the generated markdown version of the
  notebook, that cell should be tagged with `remove_cell` by selecting 
  `View -> Cell Toolbar -> Tags` and adding it. After doing this, change back
  to `None` from the same menu.
* After updating a notebook, from a terminal run the
  `<basedir>/scripts/generate_markdown.sh` script to create an updated markdown
  version of the file.

## Contributing Labs/Modules

A module should have the following directory structure:

* The base of the module should contain a README.ipynb file with a brief
  introduction to the module and links to the individual labs for each
  language translation and programming language available.
* The base of the module should contain a `C` and `Fortran` subdirectory
  containing versions of the module for C/C++ and Fortran, respectively. Each
  of these directories should contain a directory for each language
  translation provided (English, for instance).
* The base of the module should contain an `images` directory that contains
  images that will be used in common between multiple notebooks. 
* For each programming language and translation there should be a file named
  `README.ipynb` containing the actual lab instructions. A single file name
  is used in this way to simplify finding the starting point for each new
  lab.
* Each lab translation and programming language combination should have a
  `solutions` directory containing not only correct solutions, but a Makefile
  to build all contained solutions, which may be organized further into
  subdirectories if necessary. This Makefile is used to test the labs against
  future compiler releases. A Makefile outside of the solutions directory is
  optional.

