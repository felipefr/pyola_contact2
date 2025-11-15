**A fully python implementation of large displacement frictional contact**
This implementation is highly motivated by an open source 528-line MATLAB [Wang et al, 2025], that has been fully refactorised to improve efficiency and readability. 
It is itself based on a surface-to-surface formulation proposed by [Zimmerman and Ateshian, 2018; Ateshian et al, 2010], part of FEBIO FEM Solver.

@article{WANG2025103798,
title = {An open source MATLAB solver for contact finite element analysis},
journal = {Advances in Engineering Software},
volume = {199},
pages = {103798},
year = {2025},
issn = {0965-9978},
doi = {https://doi.org/10.1016/j.advengsoft.2024.103798},
url = {https://www.sciencedirect.com/science/article/pii/S0965997824002059},
author = {Bin Wang and Jiantao Bai and Shanbin Lu and Wenjie Zuo},
keywords = {Contact problems, Finite element analysis, Open source software, MATLAB implementation, Researcher development},
abstract = {Contact phenomenon widely exists in engineering, which is a high nonlinearity problem. However, the majority of open source contact finite element codes are written in C++, which are difficult for junior researchers to adopt and use. Therefore, this paper provides an open source 528-line MATLAB code and detailed interpretation for frictional contact finite element analysis considering large deformation, which is easy to learn and use by newcomers. This paper describes the contact projection, contact nodal forces and contact tangent stiffness matrices. The nonlinear equations are solved by the Newton–Raphson method. Numerical examples demonstrate the effectiveness of the MATLAB codes. The displacement, Cauchy stress and contact traction results are compared with the open-source software FEBIO.}
}

@article{Zimmerman2018Surface,
  author    = {Zimmerman, Benjamin K. and Ateshian, Gerard A.},
  title     = {A Surface-to-Surface Finite Element Algorithm for Large Deformation Frictional Contact in {febio}},
  journal   = {Journal of Biomechanical Engineering},
  volume    = {140},
  number    = {8},
  pages     = {081013-1--081013-15},
  year      = {2018},
  month     = aug,
  publisher = {ASME International},
  doi       = {10.1115/1.4040497},
  pmid      = {30003262},
  pmcid     = {PMC6056201}
}

@article{Ateshian2010Finite,
  author    = {Ateshian, Gerard A. and Maas, Steve and Weiss, Jeffrey A.},
  title     = {Finite Element Algorithm for Frictionless Contact of Porous Permeable Media Under Finite Deformation and Sliding},
  journal   = {Journal of Biomechanical Engineering},
  volume    = {132},
  number    = {6},
  pages     = {061006},
  year      = {2010},
  publisher = {ASME International},
  doi       = {10.1115/1.4001174}
}
