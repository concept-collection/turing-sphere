% Allen-Cahn on the unit sphere. One species: interfaces form and then coarsen
% until one domain swallows the sphere.
%
%   du/dt = eps2*lap(u) + u - u^3
%
% Diffusion is implicit in spherical-harmonic space, where lap is diagonal with
% eigenvalues -l(l+1); the reaction is explicit on the grid, giving one IMEX
% Euler step. Provided by the caller: synth/analys (the transforms), lam =
% l(l+1) per coefficient, noise (the seeded perturbation), and the parameters.
% Grid fields are npts x 1; spectral fields are real 2 x nlm (row 1 real part,
% row 2 imaginary), so no complex arithmetic is needed. Each function returns
% the new spectral state followed by the grid fields to display.

function [U, u] = init(noise)
  U = analys(noise);
  u = synth(U);
end

function [Un, u] = step(U, lam, eps2, dt)
  u = synth(U);
  Un = (U + dt * analys(u - u.^3)) ./ (1 + (dt * eps2) * lam);
end
