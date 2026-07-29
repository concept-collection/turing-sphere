% Brusselator reaction-diffusion on the unit sphere. Turing stripes and spots,
% from a smaller diffusivity contrast than Schnakenberg but a stiffer reaction.
%
%   du/dt = D1*lap(u) + A - (B+1)*u + u^2*v
%   dv/dt = D2*lap(v) +     B*u     - u^2*v
%
% Diffusion is implicit in spherical-harmonic space, where lap is diagonal with
% eigenvalues -l(l+1); the reaction is explicit on the grid, giving one IMEX
% Euler step. Provided by the caller: synth/analys (the transforms), lam =
% l(l+1) per coefficient, noise (the seeded perturbation), and the parameters.
% Grid fields are npts x 1; spectral fields are real 2 x nlm (row 1 real part,
% row 2 imaginary), so no complex arithmetic is needed. Each function returns
% the new spectral state followed by the grid fields to display.

function [U, V, u, v] = init(noise, A, B)
  U = analys(A + noise);
  V = analys((B / A) * ones(numel(noise), 1));
  u = synth(U);
  v = synth(V);
end

function [Un, Vn, u, v] = step(U, V, lam, A, B, D1, D2, dt)
  u = synth(U);
  v = synth(V);
  uuv = u .* u .* v;
  Un = (U + dt * analys(A - (B + 1) * u + uuv)) ./ (1 + (dt * D1) * lam);
  Vn = (V + dt * analys(B * u - uuv))           ./ (1 + (dt * D2) * lam);
end
