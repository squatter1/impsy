# author: Pascal Bercher, pascal.bercher@anu.edu.au, (3.12.2021)

PREFIX = mainfile
TEXFILE = $(PREFIX).tex
BIBFILE = $(PREFIX).aux
LATEX = pdflatex
BIBTEX = bibtex
TRASH = *.aux *.log *.dvi *.ps *.nav *.out *.snm *.backup *.bak *.toc *~ *.bbl *.blg *.toc *.fls *.brf *.fdb_latexmk

mkonline:
	latexmk -pvc -pdf $(TEXFILE)

mk:
	latexmk -pdf $(TEXFILE)

all:
	$(LATEX) -halt-on-error $(TEXFILE)
	$(BIBTEX) $(BIBFILE)
	$(LATEX) -halt-on-error $(TEXFILE)
	$(LATEX) -halt-on-error $(TEXFILE)
	$(LATEX) -halt-on-error $(TEXFILE)

quick:
	$(LATEX) -halt-on-error $(TEXFILE)

clean: clear

clear:
	rm -rf $(TRASH)
