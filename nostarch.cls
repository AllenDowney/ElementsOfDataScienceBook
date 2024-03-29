%%
%% This is file `nostarch.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% nostarch.dtx  (with options: `class')
%% 
%% IMPORTANT NOTICE:
%% 
%% For the copyright see the source file.
%% 
%% Any modified versions of this file must be renamed
%% with new filenames distinct from nostarch.cls.
%% 
%% For distribution of the original source see the terms
%% for copying and modification in the file nostarch.dtx.
%% 
%% This generated file may be distributed as long as the
%% original source files, as listed above, are part of the
%% same distribution. (The sources need not necessarily be
%% in the same archive or directory.)
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nostarch}
[2008/06/06 v1.3 Typesetting books for No Starch Press]
\newif\ifnostarch@cfonts
\nostarch@cfontstrue
\DeclareOption{cfonts}{\nostarch@cfontstrue}
\DeclareOption{nocfonts}{\nostarch@cfontsfalse}
\long\def\nostarch@size@warning#1{%
  \ClassWarning{nostarch}{Size-changing option #1 will not be
    honored}}%
\DeclareOption{8pt}{\nostarch@size@warning{\CurrentOption}}%
\DeclareOption{9pt}{\nostarch@size@warning{\CurrentOption}}%
\DeclareOption{10pt}{\nostarch@size@warning{\CurrentOption}}%
\DeclareOption{11pt}{\nostarch@size@warning{\CurrentOption}}%
\DeclareOption{12pt}{\nostarch@size@warning{\CurrentOption}}%
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\InputIfFileExists{nostarch.cfg}{%
  \ClassInfoa{nostarch}{%
    Loading configuration file nostarch.cfg}}{%
  \ClassInfo{nostarch}{%
    Configuration file nostarch.cfg is not found}}
\ProcessOptions\relax
\LoadClass[10pt]{book}
\RequirePackage[table]{xcolor}  
\RequirePackage{ifpdf, fancyhdr, fancyvrb, booktabs, graphicx,
  listings, caption, subcaption, makeidx, upquote, fontspec, pifont, textcomp, xcolor, lmodern, nameref, float, amssymb, etoolbox, setspace, marginnote, longtable}
\RequirePackage[framemethod=TikZ]{mdframed}
\RequirePackage{soulutf8}
%\RequirePackage[T1]{fontenc}  
\RequirePackage{ragged2e}
\setlength\RaggedRightParindent{\parindent}
\RaggedRight
\def\@nschapterpreamble{}
\ifnostarch@cfonts
  %\RequirePackage{nbaskerv}%
  %\RequirePackage{futurans}%
  %\RequirePackage[scaled=0.85]{thsmc}%
  %\RequirePackage{dogma}%
  %\newcommand{\dgdefault}{fdg}
  
  % ===== URLs =====
  \def\UrlFont{\em}
  
  % ===== WINGDINGS =====
  \newcommand{\wingding}[1]
    {\ding{\the\numexpr #1 + 181 \relax}} 
    
\newcommand{\codewingding}[1]
    {\vbox{\llap{\wingding{#1}\hskip 0.5em}}}
    
    
% ===== COMMANDS =====

\newcommand{\turnoffbigpara}{
    \makeatletter
    \@firstparafalse
    \makeatother
}

\newcommand{\term}[1]{\emph{#1}\index{#1}}

\newcommand{\bettertilde}{\raisebox{-1ex}{\textasciitilde}}

\newcommand{\slashslash}{%
  \raisebox{.53ex}{%
    \scalebox{.5}{%
      \rotatebox[origin=c]{18}{$\mathbf{/}$}%
    }%
  }%
}

\renewcommand{\neq}{%
  {%
   ~%
   \vphantom{=}%
   \ooalign{\kern.05em\smash{\slashslash}\hidewidth\cr$=$\cr}%
   ~%
  }%
}

% ===== SPACING =====
\sodef\sochapter{}{.150em}{.5em}{1em}
\sodef\sotitle{}{.125em}{.5em}{1em}
\sodef\sobox{}{.075em}{.5em}{1em}
\sodef\sonote{}{.200em}{.5em}{1em}



  % ===== FONTS =====
  \newfontface\dogma[
    Path=fonts/dogma/,
    UprightFont=dogmab01.ttf,
    Ligatures=TeX
]{dogma}  % Font used for chapter titles

  \newfontface\timesnewroman[
    Path=fonts/tnr/,
    UprightFont=times.ttf,
    BoldFont=timesbd.ttf,
    ItalicFont=timesi.ttf,
    BoldItalicFont=timesbi.ttf,
    Ligatures=TeX
]{timesnewroman}

  \newfontface\futurabook[
    Path=fonts/futurans/,
    UprightFont=FuturB37.ttf,
    BoldFont=FuturH79.ttf,
    ItalicFont=FutuBO08.ttf,
    BoldItalicFont=FutCBO63.ttf,
    Ligatures=TeX
]{futurabook}

  \newfontface\futuraheavy[
    Path=fonts/futurans/,
    UprightFont=FuturH79.ttf,
    Ligatures=TeX
]{futuraheavy}

  \newfontface\futuraboldc[
    Path=fonts/futurans/,
    UprightFont=futucb18.ttf,
    Ligatures=TeX
]{futuraboldc}

  \newfontface\futuraboldcoblique[
    Path=fonts/futurans/,
    UprightFont=FutCBO63.ttf,
    Ligatures=TeX
]{futuraboldcoblique}

  \newfontface\futuraoblique[
    Path=fonts/futurans/,
    UprightFont=FutuBO08.ttf,
    Ligatures=TeX
]{futuraoblique}

\setmainfont{nbaskerv}[
    Path=fonts/nbaskerv/,
    UprightFont=NewBaskervilleEF-Roman.ttf,
    BoldFont=NewBaskervilleEF-Bold.ttf,
    ItalicFont=NewBaskervilleEF-RomanIta.ttf,
    BoldItalicFont=NewBaskervilleEF-BoldIta.ttf,
    Ligatures=TeX]
    
\setmonofont[Scale=0.85,
    Path=fonts/thsmc/,
    UprightFont=ThSMCP25.ttf,
    BoldFont=ThSMCB02.ttf,
    ItalicFont=ThSMCI29.ttf,
    BoldItalicFont=TSMCBI56.ttf]{thsmc}

\setsansfont[
    Path=fonts/futurans/,
    UprightFont=futura.ttf,
    BoldFont=FuturB41.ttf,
    ItalicFont=FutuBO08.ttf,
    BoldItalicFont=FutCBO63.ttf,
    Ligatures=TeX
]{futurans}
    
    
\else
  \newcommand{\dgdefault}{\sfdefault}%
\fi
\normalsize
\setlength{\paperheight}{9.25in}
\setlength{\paperwidth}{7in}
\ifpdf\relax
  \pdfpageheight=\paperheight
  \pdfpagewidth=\paperwidth
  \pdfvorigin=1in
  \pdfhorigin=1in
\fi
\setlength\topmargin{0.625in}
\addtolength\topmargin{-1in}
\addtolength\topmargin{-12pt}
\setlength\textheight{8in}
\addtolength{\textheight}{-0.207in}
\addtolength\textheight{\topskip}
\setlength\evensidemargin{0.687in}
\addtolength{\evensidemargin}{-1in}
\addtolength{\evensidemargin}{0.833in}
\addtolength{\evensidemargin}{0.167in}
\setlength\oddsidemargin{\evensidemargin}
\setlength\textwidth{5.625in}
\addtolength\textwidth{-0.833in}
\addtolength\textwidth{-0.167in}
\setlength\parindent{0.25in}
\setlength\headheight{12pt}
\setlength\headsep{0pt}
\setlength{\footskip}{9in}
\addtolength{\footskip}{-\textheight}
\addtolength{\footskip}{0.067in}
\addtolength{\footskip}{\baselineskip}
\addtolength{\footskip}{-1in}
\widowpenalty=10000
\clubpenalty=10000
\raggedbottom
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\pagestyle{fancy}
\lhead{}
\rhead{}
\chead{}

% ===== FOOTERS =====
\fancyhfoffset[L]{1in}
\fancyfoot[RO]{\sffamily \fontsize{6pt}{6pt}%
  \fontseries{lq}\selectfont\rightmark%
  \hspace{2em} \futuraboldc \fontsize{9pt}{6pt}\selectfont\thepage}
\fancyfoot[LO]{}
\fancyfoot[CO]{}
\fancyfoot[RE]{}
\fancyfoot[CE]{}
\fancyfoot[LE]{\futuraboldc \fontsize{9pt}{6pt}%
\selectfont\thepage%
\fontseries{lq} \sffamily \fontsize{6pt}{6pt}\selectfont%
\hspace{2em}\leftmark}

% ===== SUBTITLE =====
\newcommand{\subtitle}[1]{\gdef\@subtitle{#1}}
\def\@subtitle{}

% ===== LOGO =====
\newcommand{\nostarchlogo}[1]{\gdef\@nostarchlogo{#1}}
\def\@nostarchlogo{}
\newcommand{\nostarchlocation}[1]{\gdef\@nostarchlocation{#1}}
\def\@nostarchlocation{San Francisco}

% ===== HALF TITLE =====
\def\makehalftitle{%
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \thispagestyle{empty}%
  \null%\vskip1.025in%
  \leavevmode\noindent\hspace{1.66in}%
    \begin{minipage}[t]{2.5625in}%
    \flushright
    \sffamily\bfseries\fontsize{12\p@}{13\p@}
    \MakeUppercase{\@title}%
  \end{minipage}}
  
% ===== TITLE =====
\def\maketitle{%
  \bgroup
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \thispagestyle{empty}%
  \bgroup
  \list{}{\leftmargin-1in\rightmargin\z@\labelwidth\z@%
    \labelsep\z@\listparindent\z@\parsep\z@\itemsep\z@%
    \topsep\z@\partopsep\z@}%
  \item\relax
  \centering
  \null\vskip0.855in%
  {\dogma \fontsize{36\p@}{37\p@}\selectfont
    \MakeUppercase{\sotitle{\@title}}\par}%
  \if\@subtitle\@empty\relax\else
     {\null\vskip25\p@%
     \dogma \fontsize{24\p@}{25\p@}\selectfont
     \@subtitle\par}%
  \fi
  \vspace{2.15in}%
  \vspace{-\baselineskip}%
  {\futuraheavy\fontsize{18\p@}{22\p@}\selectfont
    by \@author\par}%
  \vspace{12\p@}%
  \vspace{\baselineskip}%
  \if\@nostarchlogo\@empty\relax\else
    \includegraphics[height=1.08in]{\@nostarchlogo}\par
  \fi
  \vspace{12\p@}%
  \normalfont\@nostarchlocation\par
  \endlist
  \egroup
  \clearpage\egroup}

% ===== COPYRIGHT PAGE =====
\newenvironment{copyrightpage}{%
  \clearpage
  \bgroup
  \list{}{\leftmargin-1in\rightmargin\z@\labelwidth\z@%
    \labelsep\z@\listparindent\z@\parsep8\p@\itemsep\z@%
    \topsep\z@\partopsep\z@}%
  \item\relax
  \thispagestyle{empty}%
  \normalfont\fontsize{8\p@}{10\p@}\selectfont}{%
  \endlist
  \egroup
  \clearpage}
 
% ===== CIP BLOCK =====
\newenvironment{cipblock}{%
  \leavevmode\par\normalfont\fontsize{8\p@}{10\p@}\selectfont
  \textit{Library of Congress Cataloging-in-Publication Data}\par%
  \ttfamily\@vobeyspaces\obeylines
  \parindent\z@\parfillskip\@flushglue\parskip\z@skip}{\par\vskip8pt\relax}
  
% ===== DEDICATION PAGE =====
\newenvironment{dedicationpage}{%
  \clearpage
  \bgroup
  \list{}{\leftmargin-1in\rightmargin\z@\labelwidth\z@%
    \labelsep\z@\listparindent\z@\parsep8\p@\itemsep\z@%
    \topsep\z@\partopsep\z@}%
  \item\relax
  \thispagestyle{empty}%
  \null\vskip1.175in%
  \centering\normalfont}{%
  \endlist
  \egroup
  \clearpage}
  
\def\briefcontentsname{{Brief Contents}}
\def\contentsname{Contents in Detail}%
\AtBeginDocument{\@ifpackageloaded{babel}{%
  \addto\captionsenglish{%
    \renewcommand{\contentsname}%
    {Contents in Detail}}}{}}


% ===== BRIEF TABLE OF CONTENTS =====
\newcommand\brieftableofcontents{%
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \bgroup
  \thispagestyle{empty}%
  \markboth{\briefcontentsname}{\briefcontentsname}%
  \null\vskip0.925in%
  \list{}{\leftmargin-1.0in\rightmargin0.4in\labelwidth\z@%
    \labelsep\z@\listparindent\z@\parsep\z@\itemsep\z@%
    \topsep\z@\partopsep\z@}%
  \item\relax
    {\centering
      \dogma\fontsize{16\p@}{18\p@}\selectfont
      \MakeUppercase{\sochapter{Brief Contents}}\par}\vskip40pt%
  \@starttoc{tbc}%
  \endlist
  \egroup}
  
  % ===== SLIGHTLY BIGGER BRIEF TABLE OF CONTENTS =====
\newcommand\slightlybiggerbrieftoc{%
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \bgroup
  \thispagestyle{empty}%
  \markboth{\briefcontentsname}{\briefcontentsname}%
  \null\vskip0.925in%
  \list{}{\leftmargin-1.0in\rightmargin0.4in\labelwidth\z@%
    \labelsep\z@\listparindent\z@\parsep\z@\itemsep\z@%
    \topsep\z@\partopsep\z@}%
  \item\relax
    {\centering
      \dogma\fontsize{16\p@}{18\p@}\selectfont
      \MakeUppercase{\sochapter{Brief Contents}}\par}\vskip40pt%
  \enlargethispage{\baselineskip}
  \@starttoc{tbc}%
  \endlist
  \egroup}

% ===== TABLE OF CONTENTS =====
\renewcommand\tableofcontents{%
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \bgroup
  \thispagestyle{empty}%
  \markboth{\contentsname}{\contentsname}%
  \null\vskip0.175in%
  \list{}{\leftmargin-1.0in\rightmargin0.6in\labelwidth\z@%
    \labelsep\z@\listparindent\z@\parsep\z@\itemsep\z@%
    \topsep\z@\partopsep\z@}%
  \item\relax
    {\centering
      \dogma\fontsize{16\p@}{18\p@}\selectfont
      \MakeUppercase{\sochapter{Contents in Detail}}\par}\vskip18pt%
  \@starttoc{toc}%
  \endlist
  \egroup}
\renewcommand\@dotsep{1.5}

% ===== PART IN ToC =====
\renewcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{28\p@}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \sffamily\bfseries\fontsize{12\p@}{14\p@}\selectfont
       #1}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}
  
% ===== CHAPTER IN ToC =====
\renewcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 20\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode
      \sffamily \bfseries \fontsize{10\p@}{12\p@}{#1} 
      \nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \vskip4\p@\penalty\@highpenalty
    \endgroup
  \fi}
  

% ===== SECTION WIDTHS =====
\newlength{\sectionnumberwidth}
\settowidth{\sectionnumberwidth}{%
  \futurabook\fontsize{9\p@}{11\p@}\selectfont
  99.99}
\newlength{\boldsectionnumberwidth}
\settowidth{\boldsectionnumberwidth}{%
  \futuraheavy\fontsize{9\p@}{11\p@}\selectfont
  99.99}
\newlength{\subsectionnumberwidth}
\settowidth{\subsectionnumberwidth}{%
  \futurabook\fontsize{9\p@}{11\p@}\selectfont
  99.99.99}
  
% ===== SECTION IN ToC =====
\renewcommand{\l@section}[2]{\@dottedtocline{1}{\z@}{%
    \sectionnumberwidth}{%
    \futurabook\fontsize{9\p@}{11\p@}\selectfont#1}{%
    \futurabook\fontsize{9\p@}{11\p@}\selectfont#2}}
    
% ===== SUBSECTION IN ToC =====
\renewcommand{\l@subsection}[2]{\@dottedtocline{2}{0.5in}%
  {\subsectionnumberwidth}{%
    \futurabook\fontsize{9\p@}{11\p@}\selectfont#1}{%
     \futurabook\fontsize{9\p@}{11\p@}\selectfont#2}}
     
% ===== BOLD SECTION IN ToC =====
\newcommand*\l@boldsection[2]{{\bfseries\@dottedtocline{1}{\z@}{%
    \boldsectionnumberwidth}{%
    \futuraheavy\fontsize{9\p@}{11\p@}\selectfont#1}{%
    \futuraheavy\fontsize{9\p@}{11\p@}\selectfont#2}}}

% ===== PART IN BRIEF ToC =====
\newcommand*\l@bpart[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{14\p@}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \futuraheavy\fontsize{9\p@}{14\p@}\selectfont
       #1}\par\vskip7\p@%
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}
  
% ===== CHAPTER IN BRIEF ToC =====
\renewcommand{\@pnumwidth}{17pt}
\newcommand*\l@bchapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode
      \futurabook\fontsize{9\p@}{14\p@}\selectfont
      #1\nobreak\leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{.}\mkern \@dotsep
        mu$}\hfill
      \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
  
\newcommand*\l@blanktbcchapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode
      \futurabook\fontsize{9\p@}{14\p@}\selectfont
      #1\nobreak\leaders\hbox{$\m@th
        \mkern \@dotsep mu\hbox{}\mkern \@dotsep
        mu$}\hfill
      \nobreak\hb@xt@\@pnumwidth{\hss}\par
      \penalty\@highpenalty
    \endgroup
  \fi}

  
% ===== MAIN MATTER =====
\renewcommand\mainmatter{%
  \clearpage
  {\thispagestyle{empty}\cleardoublepage}%
  \@mainmattertrue
  \pagenumbering{arabic}}
  
% ===== BACK MATTER =====
\renewcommand\backmatter{%
  \clearpage
  {\thispagestyle{empty}\cleardoublepage}%
  \@mainmatterfalse%
  \phantomsection%
  \addcontentsline{tbc}{blanktbcchapter}{}%
  }
  
\newcommand{\apptoc}{%
    \renewcommand{\chaptername}{Appendix}%
    \phantomsection%
    \addcontentsline{tbc}{blanktbcchapter}{}%
}
  
%\setcounter{secnumdepth}{0}
\def\sectionmark#1{}%
\newif\if@firstpara
\@firstparafalse
\def\@chapterart{}
\def\secdef#1#2{\@ifstar{\@dblarg{#2}}{\@dblarg{#1}}}

% ===== PART =====
\renewcommand\part{%
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \everypar{}%
  \markboth{}{}%
  \def\@chapterart{}%
  \thispagestyle{empty}%
  \null\vskip0.715in%
  \secdef\@part\@spart}
\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\MakeUppercase{\partname}
        \thepart\protect\linebreak\penalty\@highpenalty\MakeUppercase{#1}}%
      \addcontentsline{tbc}{bpart}{\MakeUppercase{\partname~\thepart: \futuraheavy #1}}%
    \else
      \addcontentsline{toc}{part}{\MakeUppercase{#1}}%
      \addcontentsline{tbc}{bpart}{\MakeUppercase{ \futuraheavy #1}}%
    \fi
    {\centering
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >-2\relax
       \futuraboldc\fontsize{80pt}{80pt}\fontseries{bc}\selectfont
       \MakeUppercase{\partname\nobreakspace\thepart}%
       \par
       \vskip 0.45in%
     \fi
     \dogma \fontsize{16pt}{16pt}\fontseries{br}\selectfont
     \MakeUppercase{\sochapter{#2}}\par}%
    \@endpart}
\def\@spart[#1]#2{%
  \addcontentsline{toc}{part}{\MakeUppercase{#1}}%
  \addcontentsline{tbc}{bpart}{\MakeUppercase{#1}}%
    {\centering
      \dogma\fontsize{16pt}{16pt}\fontseries{br}\selectfont
     \MakeUppercase{\sochapter{#2}}\par}%
    \@endpart}
\def\@endpart{%
  \@afterindentfalse
  \vskip49\p@%
   \@firstparatrue
   \@afterheading}
   
% ===== CHAPTER =====
\def\chapter{%
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \everypar{}%
  \thispagestyle{empty}%
  \global\@topnum\z@
  \@afterindentfalse
  \gdef\@chapterart{}%
  \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{%
  \vspace*{0.622in}
  \ifnum \c@secnumdepth > \m@ne
  \refstepcounter{chapter}%
  \typeout{Chapter:\thechapter\space\@currentlabel}%
  \markboth{\@chapapp~\thechapter}{#1}%
  \typeout{\@chapapp\space\thechapter.}%
  \addcontentsline{toc}{chapter}{\thechapter%
    \protect\linebreak\penalty\@highpenalty
    \MakeUppercase{#1}}%
  \addcontentsline{tbc}{bchapter}{%
    \chaptername~\thechapter:~#1}%
  {\centering
    \futuraboldc\fontsize{120pt}{120pt}
    \selectfont
    \thechapter\par\nobreak\vskip27pt}%
  \else
  \addcontentsline{toc}{chapter}{%
    \MakeUppercase{#1}}%
  \addcontentsline{tbc}{bchapter}{#1}%
  \markboth{#1}{#1}%
  \fi
  {\centering
    \dogma\fontsize{16pt}{18pt}\selectfont
    \MakeUppercase{\sochapter{#2}}\par}\nobreak\vskip25pt\relax%
  \@firstparatrue
  \@afterheading}%
\def\@schapter[#1]#2{%
  \vspace*{2.747in}
  {\centering
    \phantomsection
    \addcontentsline{toc}{chapter}{%
      \MakeUppercase{#1}}%
    \addcontentsline{tbc}{bchapter}{#1}%
    \markboth{#1}{#1}%
    {\fontsize{16pt}{18pt}\selectfont
      \MakeUppercase{\dogma \sochapter{#2}}\par}}\nobreak\vskip25pt\relax%
   \@firstparatrue
   \@afterheading}%
\def\@afterheading{%
  \@nobreaktrue
  \everypar{%
    \if@nobreak
      \@nobreakfalse
      \clubpenalty \@M
       \setbox\z@\lastbox
    \else
      \clubpenalty \@clubpenalty
      \everypar{}%
    \fi
      \if@firstpara
       \rule{\z@}{49\p@}%
       \ifx\@chapterart\@empty%
       \else
         \makebox[0pt][r]{\raisebox{-0.5in}[0pt][0pt]{%
            \@chapterart\hspace{0.21in}}}%
        \parshape=5
        0.963in 3.622in
        0.913in 3.672in
        0.813in 3.772in
        0.563in 4.022in
        0in \textwidth
      \fi
      \fontsize{14pt}{16.5pt}\selectfont%
      \parskip=3pt
      \else
        \parskip=0pt
        \normalsize\selectfont
       \fi
    \@firstparafalse}}%
    
 % ===== CHAPTER WITH QUOTE =====
\def\chapterwithquote{%
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \everypar{}%
  \thispagestyle{empty}%
  \global\@topnum\z@
  \@afterindentfalse
  \gdef\@chapterart{}%
  \secdef\@chapterwithquote\@schapterwithquote}
\def\@chapterwithquote[#1]#2#3#4{%
  \vspace*{0.622in}
  \ifnum \c@secnumdepth > \m@ne
  \refstepcounter{chapter}%
  \typeout{Chapter:\thechapter\space\@currentlabel}%
  \markboth{\@chapapp~\thechapter}{#1}%
  \typeout{\@chapapp\space\thechapter.}%
  \addcontentsline{toc}{chapter}{\thechapter%
    \protect\linebreak\penalty\@highpenalty
    \MakeUppercase{#1}}%
  \addcontentsline{tbc}{bchapter}{%
    \chaptername~\thechapter:~#1}%
  {\centering
    \futuraboldc\fontsize{120pt}{120pt}
    \selectfont
    \thechapter\par\nobreak\vskip27pt}%
  \else
  \addcontentsline{toc}{chapter}{%
    \MakeUppercase{#1}}%
  \addcontentsline{tbc}{bchapter}{#1}%
  \markboth{#1}{#1}%
  \fi
  {\centering
    \dogma\fontsize{16pt}{18pt}\selectfont
    \MakeUppercase{\sochapter{#2}}\par\vspace{.1in}\normalfont\fontsize{9pt}{11pt}\selectfont\textit{#3}\par\textit{\textemdash#4}\par}\nobreak\vskip25pt\relax%
  \@firstparatrue
  \@afterheading}%
\def\@schapterwithquote[#1]#2#3#4{%
  \vspace*{2.747in}
  {\centering
    \phantomsection
    \addcontentsline{toc}{chapter}{%
      \MakeUppercase{#1}}%
    \addcontentsline{tbc}{bchapter}{#1}%
    \markboth{#1}{#1}%
    {\dogma\fontsize{16pt}{18pt}\selectfont
    \MakeUppercase{\sochapter{#2}}\par\vspace{.1in}\normalfont\fontsize{9pt}{11pt}\selectfont\textit{#3}\par\textit{\textemdash#4}\par}}\nobreak\vskip25pt\relax%   
    \@firstparatrue
   \@afterheading}%


% ===== CHAPTER ART =====
\newcommand{\chapterart}[1]{\gdef\@chapterart{#1}}%
\newcommand{\chapterartfile}[1]{\gdef\@chapterart{%
    \includegraphics[width=1.264in]{#1}}}%

% ===== SECTION =====
\renewcommand{\section}{\everypar{}%
  \@startsection{section}{1}{-0.75in}{16pt}{6pt}{%
    \sffamily \bfseries \fontsize{12pt}{16pt}\selectfont}}%
    
% ===== SUBSECTION =====
\renewcommand{\subsection}{\everypar{}%
  \@startsection{subsection}{2}{0pt}{14pt}{4pt}{%
    \futuraboldcoblique %
    \fontsize{12pt}{17pt}\selectfont}}%
    
% ===== SUBSUBSECTION =====
\renewcommand{\subsubsection}{\everypar{}%
  \@startsection{subsubsection}{3}{0pt}{10pt}{2pt}{%
  \futuraboldc \fontsize{10pt}{14pt}%
    \selectfont}}%
    
% ===== LISTS =====
\setlength{\leftmargin}{0.25in}
\setlength{\leftmargini}{0.25in}
\setlength{\leftmarginii}{0.25in}
\setlength{\leftmarginiii}{0.25in}
\setlength{\leftmarginiv}{0.25in}
\setlength{\rightmargin}{0pt}
\def\@listi{\leftmargin\leftmargini
  \labelwidth\leftmargini
  \advance\labelwidth-\labelsep\relax
  \parsep0\p@\relax
  \topsep6\p@\relax
  \itemsep4\p@\relax}
\let\@listI\@listi
\def\@listii{\leftmargin\leftmarginii
  \labelwidth\leftmarginii
  \advance\labelwidth-\labelsep
  \parsep0\p@\relax
  \topsep0\p@\relax
  \itemsep0\p@\relax}
\def\@listiii{\leftmargin\leftmarginiii
  \labelwidth\leftmarginiii
  \advance\labelwidth-\labelsep
  \parsep0\p@\relax
  \topsep0\p@\relax
  \itemsep0\p@\relax}
\def\@listiv{\leftmargin\leftmarginiv
  \labelwidth\leftmarginiv
  \advance\labelwidth-\labelsep
  \parsep0\p@\relax
  \topsep0\p@\relax
  \itemsep0\p@\relax}
\def\@listv{\leftmargin\leftmarginv
  \labelwidth\leftmarginv
  \advance\labelwidth-\labelsep
  \parsep0\p@\relax
  \topsep0\p@\relax
  \itemsep0\p@\relax}
\def\@listvi{\leftmargin\leftmarginvi
  \labelwidth\leftmarginvi
  \advance\labelwidth-\labelsep
  \parsep0\p@\relax
  \topsep0\p@\relax
  \itemsep0\p@\relax}

% ===== ITEMIZE =====
\def\itemize{%
  \ifnum \@itemdepth >\thr@@\@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \expandafter
    \list
      \csname\@itemitem\endcsname
      {\def\makelabel##1{##1}}%
  \fi}
  
% ===== ENUMERATE =====
\def\enumerate{%
  \ifnum \@enumdepth >\thr@@\@toodeep\else
    \advance\@enumdepth\@ne
    \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
      \expandafter
      \list
        \csname label\@enumctr\endcsname
        {\usecounter\@enumctr\def\makelabel##1{##1}}%
  \fi}
  
      
% ===== LIST PLAIN ===== 
\renewenvironment{description}%
  {\list{}{\labelwidth=0pt \leftmargin=18pt
   \let\makelabel\descriptionlabel}}
  {\endlist}
        
% ===== QUOTES =====
\renewenvironment{quotation}{%
  \list{}{\listparindent\parindent\relax
    \itemindent\listparindent\relax
    \rightmargin0.5in\relax
    \leftmargin0.5in\relax}%
    \item\fontsize{9pt}{11pt}\selectfont}{\endlist}
\renewenvironment{quote}{%
  \list{}{\listparindent0pt\relax
    \itemindent\listparindent\relax
    \rightmargin0.5in\relax
    \leftmargin0.5in\relax}%
    \item\fontsize{9pt}{11pt}\selectfont}{\endlist}

% ===== NOTE =====
\newenvironment{note}{%
  \list{\makebox[0pt][r]{\dogma \fontsize{7.5pt}{12pt}
      \colorbox{black}{\textcolor{white}{\sonote{NOTE}}}\hspace{2em}}}{\listparindent0pt\relax
    \topsep9\p@\relax
    \itemindent0\p@\relax
    \rightmargin0\p@\relax
    \leftmargin0\p@\relax
    \labelwidth0\p@\relax
    \labelsep0\p@}%
    \item\itshape}{\vspace{-3pt}\endlist}
    
% ===== FOOTNOTE =====
\renewcommand\footnoterule{%
  \kern6\p@
  \hrule height 0.25pt depth 0pt width 1in
  \kern2\p@}
\long\def\@footnotetext#1{\insert\footins{%
    \reset@font\footnotesize
    \interlinepenalty\interfootnotelinepenalty
    \splittopskip\footnotesep
    \splitmaxdepth \dp\strutbox \floatingpenalty \@MM
    \hsize\columnwidth \@parboxrestore
    \protected@edef\@currentlabel{%
       \csname p@footnote\endcsname\@thefnmark
    }%
    \color@begingroup
      \@makefntext{%
        \rule\z@{13.5pt}\ignorespaces#1}%
    \color@endgroup}}%
\setlength{\skip\footins}{20\p@ \@plus 4\p@ \@minus 0\p@}
\renewcommand\@makefntext[1]{%
    \parindent 0\p@%
    \RaggedRightParindent0\p@%
    \noindent
    \@makefnmark\space#1}
    
\renewcommand\@makefntext[1]{%
\@thefnmark.~#1}

% ===== FIGURE NUMBERING =====
\renewcommand \thefigure
     {\ifnum \c@chapter>\z@ \thechapter-\fi \@arabic\c@figure}
     
% ===== TABLE NUMBERING =====
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \thechapter-\fi \@arabic\c@table}


% ===== LISTING =====
\AtBeginDocument{%
  \renewcommand \thelstlisting
       {\ifnum \c@chapter>\z@ \thechapter-\fi \@arabic\c@lstlisting}%
}

\lstset{
    frame = lines, 
    showstringspaces=false,
    tabsize=4,
    title=\lstname,
    numbers = none,
    caption={},
    literate={\ }{{\ }}1{"}{\textquotedbl}1,  
    escapeinside={(@}{@)},
    captionpos=b,
    basicstyle=\ttfamily,
    columns=fullflexible,
    moredelim=[is][\bfseries]{***}{***},
    moredelim=[is][\itshape]{___}{___},
    moredelim=[is][\bfseries\itshape]{&&&}{&&&},
    moredelim=[is][\color{gray}]{@@@}{@@@},
    aboveskip=8pt,
}

\lstnewenvironment
    {codewide}[1][]
    {
    \renewcommand{\codewingding}[1]
    {\ding{\the\numexpr ##1 + 181 \relax}} 
   \captionsetup[lstlisting]{font=nostarchfigfont,singlelinecheck=off,
  aboveskip=4pt, belowskip=-12pt, margin=-70pt}
    \lstset{
    frame = lines, 
    showstringspaces=false,
    tabsize=4,
    title=\lstname,
    numbers = none,
    literate={\ }{{\ }}1{"}{\textquotedbl}1,  
    escapeinside={(@}{@)},
    captionpos=b,
    caption={},
    basicstyle=\ttfamily,
    columns=fullflexible,
    moredelim=[is][\bfseries]{***}{***},
    moredelim=[is][\itshape]{___}{___},
    moredelim=[is][\bfseries\itshape]{&&&}{&&&},
    moredelim=[is][\color{gray}]{@@@}{@@@},
    xleftmargin=-70pt,
    aboveskip=8pt,
    #1
    }}
    {
    \renewcommand{\codewingding}[1]
    {\vbox{\llap{\wingding{#1}\hskip 0.5em}}}
    \captionsetup[lstlisting]{font=nostarchfigfont,singlelinecheck=off,
  aboveskip=4pt, belowskip=0pt, margin=0pt}}


% ===== CAPTIONS =====
\DeclareCaptionFont{nostarchfigfont}{\sffamily \itshape \fontsize{8.5pt}{9pt}\selectfont}
\DeclareCaptionFont{nostarchtabfont}{\futurabook \fontsize{9pt}{9pt}\selectfont}
\captionsetup[figure]{font=nostarchfigfont,singlelinecheck=off,
  aboveskip=8pt,belowskip=-8pt}
\captionsetup[lstlisting]{font=nostarchfigfont,singlelinecheck=off,
  aboveskip=4pt, belowskip=0pt}
\captionsetup[table]{font=nostarchtabfont,singlelinecheck=off,
  aboveskip=0pt,belowskip=4pt, labelfont=bf}
  
  
% ===== ALT LST =====
\newcommand{\captionlst}[1]{{\ttfamily \itshape \fontsize{10pt}{9pt}\selectfont #1}}
\newcommand{\boxlst}[1]{{\ttfamily \fontsize{10pt}{9pt}\selectfont #1}}
\newcommand{\tablelst}[1]{{\ttfamily \fontsize{10pt}{9pt}\selectfont #1}}


  
% ===== TABLES =====
\def\tbfont{%
  \futurabook\fontsize{8.5pt}{10pt}\selectfont}
\def\thfont{%
  \futuraheavy\fontsize{9pt}{10pt}\selectfont}
  
\AtBeginEnvironment{tabular}{\tbfont}
 
\heavyrulewidth=1\p@
\lightrulewidth=1\p@
\cmidrulewidth=1\p@
\def\bottomrule{\noalign{\ifnum0=`}\fi
  \@aboverulesep=\aboverulesep
  \global\@belowrulesep=\belowbottomsep
  \global\@thisruleclass=\@ne
  \@ifnextchar[{\@BTrule}{\@BTrule[\lightrulewidth]}}
\belowrulesep=0.7ex
\belowbottomsep=0.65pt
\aboverulesep=0.7ex
\abovetopsep=0.65pt

\newenvironment{tablewide}[1][]
    {
        \captionsetup[table]{font=nostarchtabfont,singlelinecheck=off,
  aboveskip=0pt,belowskip=4pt, labelfont=bf, margin=-70pt}
        \begin{table}[#1]
    }
    {
        \end{table}
        \captionsetup[table]{font=nostarchtabfont,singlelinecheck=off,
  aboveskip=0pt,belowskip=4pt, labelfont=bf, margin=0pt}
    }
    
\newenvironment{tabularwide}[1]
    {
        \hspace*{-72pt}
        \begin{tabular}{#1}
    }
    {
        \end{tabular}
    }
    
    \newenvironment{figurewide}[1][]
    {
        \captionsetup[figure]{font=nostarchfigfont,singlelinecheck=off,
  aboveskip=8pt,belowskip=-8pt, margin=-70pt}
        \begin{figure}[#1]
        \hspace*{-70pt}
    }
    {
        \end{figure}
        \captionsetup[figure]{font=nostarchfigfont,singlelinecheck=off,
  aboveskip=8pt,belowskip=-8pt}
    }

\DefineVerbatimEnvironment{Code}{Verbatim}{frame=lines,framerule=0.25pt, commandchars=\\\{\}}
\FV@AddToHook\FV@ListParameterHook{\vspace{-6pt}}
\VerbatimFootnotes

% ===== BOX =====
\definecolor{nspgray}{cmyk}{0, 0, 0, 0.1}
\mdfdefinestyle{nspbox}{%
    linecolor=black,
    outerlinewidth=1pt,
    roundcorner=5pt,
    innertopmargin=\baselineskip,
    innerbottommargin=\baselineskip,
    innerrightmargin=18pt,
    innerleftmargin=18pt,
    backgroundcolor=nspgray,
    }
    
\mdfdefinestyle{custombox}{%
    linecolor=black,
    outerlinewidth=1pt,
    roundcorner=5pt,
    innertopmargin=\baselineskip,
    innerbottommargin=\baselineskip,
    innerrightmargin=18pt,
    innerleftmargin=18pt,
    }
    
%1p6
\newenvironment{nspbox}[1]
{\begin{mdframed}[style=nspbox]
    \begin{center} 
    {\dogma \fontsize{9pt}{15pt} \MakeUppercase{\sobox{#1}}}
    \end{center}
    \begin{flushright}
    \begin{flushleft}
    \parskip=.5\baselineskip
    \tbfont
}
{\end{flushleft}\end{flushright}
\end{mdframed}}

\newenvironment{nspboxcontinued}
{\begin{mdframed}[style=nspbox]
    \begin{flushright}
    \begin{flushleft}
    \parskip=.5\baselineskip
    \tbfont
}
{
\end{flushleft}
\end{flushright}
\end{mdframed}}

\newenvironment{custombox}[1]
{\begin{mdframed}[style=custombox]
    \begin{center} 
    {\dogma \fontsize{9pt}{15pt} \MakeUppercase{\sobox{#1}}} 
    \end{center}
    \begin{flushright}
    \begin{flushleft}
    \parskip=.5\baselineskip
    \tbfont
}
{\end{flushleft}\end{flushright}
\end{mdframed}}

\newenvironment{customboxcontinued}
{\begin{mdframed}[style=custombox]
    \begin{flushright}
    \begin{flushleft}
    \parskip=.5\baselineskip
    \tbfont
}
{
\end{flushleft}
\end{flushright}
\end{mdframed}}

\newcommand{\continued}[0]{%
    {\begin{flushright}%
    \emph{(continued)}%
    \end{flushright}}%
}%

% ===== KEYCAP =====
% Taken from: https://tex.stackexchange.com/questions/55664/fake-small-caps-with-xetex-fontspec/225078
\newcommand\keycap[1]{\fauxschelper#1 \relax\relax}
\def\fauxschelper#1 #2\relax{%
  \fauxschelphelp#1\relax\relax%
  \if\relax#2\relax\else\ \fauxschelper#2\relax\fi%
}
\def\Hscale{.83}\def\Vscale{.72}\def\Cscale{1.00}
\def\fauxschelphelp#1#2\relax{%
  \ifnum`#1>``\ifnum`#1<`\{\scalebox{\Hscale}[\Vscale]{\uppercase{#1}}\else%
    \scalebox{\Cscale}[1]{#1}\fi\else\scalebox{\Cscale}[1]{#1}\fi%
  \ifx\relax#2\relax\else\fauxschelphelp#2\relax\fi}
  
\newcounter{project}
\stepcounter{project}

% ===== PROJECT HEAD =====
\newcommand{\projecthead}[1]{
    \vspace{16pt}%
    \noindent%
    \hspace{-.75in}%
    \addcontentsline{toc}{boldsection}{Project~\#\theproject: #1}%
     {\sffamily\bfseries\fontsize{12pt}{16pt}\selectfont 
        \colorbox{black}{\textcolor{white}{Project~\#\theproject: #1}}}%
    \vspace{6pt}%
    \@afterindentfalse\@afterheading%
    \stepcounter{project}%
}%
 
 % ===== MENU ARROW =====
 \newcommand\menuarrow{\raisebox{.2ex}{\small $\blacktriangleright$}}
 
  % ===== PLAIN LIST =====
 \newenvironment{plainlist}{%
   \renewcommand{\descriptionlabel}[1]{\hspace{\labelsep}\textbf{##1}} % Add em space to description label
   \begin{description}%
}{%
    \renewcommand{\descriptionlabel}[1]{\hspace{\labelsep}\textbf{##1 \quad}} % Add em space to description label
   \end{description}%
}

% ===== MARGIN FILENAME =====
\renewcommand*{\marginfont}{\sffamily\itshape\fontsize{8.5}{10}\selectfont}
\renewcommand*{\marginnotevadjust}{-\baselineskip}
\newcommand{\marginwingding}[1]{{\fontsize{10pt}{12pt}\selectfont \wingding{#1}}}


% Code below makes the margin note always on the left, taken from: https://tex.stackexchange.com/questions/472728/marginnote-always-on-left-side-of-the-page
\long\def\@mn@@@marginnote[#1]#2[#3]{%
  \begingroup
    \ifmmode\mn@strut\let\@tempa\mn@vadjust\else
      \if@inlabel\leavevmode\fi
      \ifhmode\mn@strut\let\@tempa\mn@vadjust\else\let\@tempa\mn@vlap\fi
    \fi
    \@tempa{%
      \vbox to\z@{%
        \vss
        \@mn@margintest
        \if@reversemargin\if@tempswa
            \@tempswafalse
          \else
            \@tempswatrue
        \fi\fi

          \llap{%
            \vbox to\z@{\kern\marginnotevadjust\kern #3
              \vbox to\z@{%
                \hsize\marginparwidth
                \linewidth\hsize
                \kern-\parskip
                %\mn@parboxrestore
                \marginfont\raggedleftmarginnote\strut\hspace{\z@}%
                \ignorespaces#1\endgraf
                \vss
              }%
              \vss
            }%
            \if@mn@verbose
              \PackageInfo{marginnote}{xpos seems to be \@mn@currxpos}%
            \fi
            \begingroup
              \ifx\@mn@currxpos\relax\else\ifx\@mn@currpos\@empty\else
                  \kern\@mn@currxpos
              \fi\fi
              \ifx\@mn@currpage\relax
                \let\@mn@currpage\@ne
              \fi
              \if@twoside\ifodd\@mn@currpage\relax
                  \kern-\oddsidemargin
                \else
                  \kern-\evensidemargin
                \fi
              \else
                \kern-\oddsidemargin
              \fi
              \kern-1in
            \endgroup
            \kern\marginparsep
          }%
      }%
    }%
  \endgroup
}

% ===== BIBLIOGRAPHY =====
\AtBeginDocument{
  \def\bibsection{%
    \clearpage
    {\pagestyle{empty}\cleardoublepage}%
    \everypar{}%
    \thispagestyle{empty}%
    \global\@topnum\z@
    \@afterindentfalse
    \gdef\@chapterart{}%
    \@nschapterpreamble
    \vspace*{0.622in}%
    \addcontentsline{toc}{chapter}{\MakeUppercase{\bibname}}%
    \addcontentsline{tbc}{bchapter}{\bibname}%
    {\centering
      {\dogma\fontsize{16pt}{20pt}\selectfont
        \MakeUppercase{\sochapter{\bibname}}\par}}\nobreak\vskip25pt\relax%
    \@mkboth{\bibname}{\bibname}}}%
\renewenvironment{thebibliography}[1]{%
  \bibsection
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
      
% ===== INDEX =====
\renewenvironment{theindex}{%
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \everypar{}%
  \thispagestyle{empty}%
  \global\@topnum\z@
  \@afterindentfalse
  \gdef\@chapterart{}%
  \if@twocolumn
  \@restonecolfalse
  \else
  \@restonecoltrue
  \fi
  \twocolumn[%
  \@nschapterpreamble
  \vspace*{0.622in}%
  \addcontentsline{toc}{chapter}{\MakeUppercase{\indexname}}%
  \addcontentsline{tbc}{bchapter}{\indexname}%
  {\centering
    {\dogma%
      \fontsize{16pt}{20pt}\selectfont
      \MakeUppercase{\sochapter{\indexname}}\par}}%
  \nobreak\vskip25pt\relax]%
  \@mkboth{\indexname}%
  {\indexname}%
  \parindent\z@
  \parskip\z@ \@plus .3\p@\relax
  \columnseprule \z@
  \columnsep 0.25in%
  \let\item\@idxitem}
{\if@restonecol\onecolumn\else\clearpage\fi}
\renewcommand\@idxitem{%
  \par\hangindent0.45in\normalfont\fontsize{9.5\p@}{11.5\p@}\selectfont
  \raggedright}
\renewcommand\subitem{\@idxitem \hspace*{0.15in}}
\renewcommand\subsubitem{\@idxitem \hspace*{0.3in}}
\renewcommand\indexspace{\par \vskip 14\p@\relax}
\newcommand\indexgroup[1]{{ \futuraboldc\fontsize{11\p@}{13\p@}\selectfont#1}\par\vskip4\p@}
    
% ===== UPDATES =====
\def\updatesname{Updates}
\newcommand\updatespage{%
  \clearpage
  {\pagestyle{empty}\cleardoublepage}%
  \everypar{}%
  \thispagestyle{empty}%
  \global\@topnum\z@
  \gdef\@chapterart{}%
  \vspace*{1.875in}%
  {\centering
   \dogma%
    \fontsize{16pt}{20pt}\selectfont
    \MakeUppercase{\sochapter{\updatesname}}\par\nobreak\vskip40\p@}%
  \@nobreaktrue
  \everypar{%
    \if@nobreak
    \@nobreakfalse
    \clubpenalty \@M
    \setbox\z@\lastbox
    \else
    \clubpenalty \@clubpenalty
    \everypar{}%
    \fi}}
    
% ===== COLOPHON =====
\def\colophonname{Colophon}
\newcommand\colophon{%
  \everypar{}%
  \global\@topnum\z@
  \gdef\@chapterart{}%
  \vspace*{0.875in}%
  %{\centering
    %\dogma%
    %\fontsize{16pt}{20pt}\selectfont
    %\MakeUppercase{\sochapter{\colophonname}}\par\nobreak\vskip40\p@}%
  \@nobreaktrue
  \everypar{%
    \if@nobreak
    \@nobreakfalse
    \clubpenalty \@M
    \setbox\z@\lastbox
    \else
    \clubpenalty \@clubpenalty
    \everypar{}%
    \fi}\par
    The fonts used in \emph{\@title} are New Baskerville, Futura, The
    Sans Mono Condensed, and Dogma.  The book was typeset with
    \LaTeXe{} package
    \texttt{nostarch} by Boris Veytsman
    \emph{(\csname ver@nostarch.cls\endcsname).}\par}
\endinput
%%
%% End of file `nostarch.cls'.
