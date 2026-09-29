"""Build cl/main.tex (Computational Linguistics, clv2025) from ../main.tex.

    python cl/gerar_cl.py

The body is taken verbatim from the generic manuscript, so the two cannot diverge;
only the preamble, paths, bibliography style and appendix heading differ.
"""
import pathlib, re
AQUI = pathlib.Path(__file__).resolve().parent
PRE = r"""% ============================================================================
% papers/attention-selective-ner/cl/main.tex -- Computational Linguistics version
% ----------------------------------------------------------------------------
% Target (author decision, 2026-09-28): Computational Linguistics, short paper
% (15-25 journal pages of main content, excluding acknowledgements, references
% and appendices; abstract 150-250 words; NOT double-blind: names and
% affiliations on page one). Class: clv2025.cls (downloaded 2026-09-28 from
% submissions.cljournal.org/stylefiles). Body identical to ../main.tex.
% Every result is a macro from ../numeros.tex. Do not type a result here.
% ============================================================================
\documentclass[shortpaper]{clv2025}
\jvol{vv}
\jnum{nn}
\jyear{2026}
\dochead{Short Paper}
\runningtitle{Attention Is Not Not Geometry}
\runningauthor{Souza, Espíndola, and Cerqueira}

\usepackage{amsmath,amssymb,mathtools}
\usepackage{graphicx}
\graphicspath{{../img/}}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{float}
\usepackage[dvipsnames]{xcolor}
\usepackage[capitalise]{cleveref}
% the class sets an 8-bit Palatino; map the few characters it lacks
\usepackage{newunicodechar}
\newunicodechar{ć}{\'c}
\newunicodechar{Š}{\v{S}}
\newunicodechar{—}{---}
\newunicodechar{–}{--}
\newunicodechar{−}{\ensuremath{-}}

\newcommand{\gliner}{GLiNER}
\newcommand{\aurc}{\textsc{aurc}}
\newcommand{\authornote}[1]{\textcolor{BrickRed}{\textbf{[Open decision:} #1\textbf{]}}}
\input{../numeros}
\input{../numeros_nulo}
\input{../publicacao}

\begin{document}
\title{Attention Is Not Not Geometry}
\author{Anaximandro Souza\thanks{Corresponding author}$^{,1}$, Rogério Pinto Espíndola$^{1}$, Renato Cerqueira$^{2}$}
\affilblock{
    \affil{PEC/COPPE, Universidade Federal do Rio de Janeiro (UFRJ)\\\quad \email{anaximandro.souza@coc.ufrj.br}}
    \affil{PUC-Behring Institute for AI}
}
\maketitle

"""
src = (AQUI.parent / "main.tex").read_text(encoding="utf-8")
body = src[src.index("\\begin{abstract}"):]
body = body.replace("img/fig_", "fig_")
body = body.replace("\\input{tab_estratos}", "\\input{../tab_estratos}")
body = body.replace("\\input{tab_estratos_decoder}", "\\input{../tab_estratos_decoder}")
body = body.replace("\\bibliographystyle{plainnat}\n\\bibliography{references}",
                    "\\bibliographystyle{compling}\n\\bibliography{../references}")
body = body.replace("\\section{Pre-registration and analysis protocol}", "\\appendixsection{Pre-registration and analysis protocol}")
body = re.sub(r"([.!?]\s+)Appendix~\\ref\{app:protocol\}", r"\1The Appendix", body)
body = body.replace("(Appendix~\\ref{app:protocol})", "(Appendix)").replace("Appendix~\\ref{app:protocol}", "the Appendix")
body = re.sub(r"\\paragraph\{([^{}]*?)\.\}", r"\\paragraph{\1}", body)   # the class adds the period
body = re.sub(r"\\paragraph\{([^{}]*?)\.''\}", r"\\paragraph{\1''}", body)
(AQUI / "main.tex").write_text(PRE + body, encoding="utf-8")
print("cl/main.tex written")
