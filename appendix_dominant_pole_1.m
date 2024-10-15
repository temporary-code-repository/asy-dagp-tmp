%% Behaviour of the dominant pole by changing the ratio of the divided network
clc
clear
close

syms z

%% Select the parameters
eta   = 0.5;  
gamma = 0.95; 
L     = 1.;  
mu    = 0.1; 
zeta  = 0.1; 
beta  = .00;
rho   = 0.5;
alpha = 0.5;
pwrw  = 0.25;     %% Weight power parameter, related to $\iota$ parameter in the paper. 

thr_imag = 0.1;   %% Threshold set to compensate for numerical errors resulting from symbolic and complex multiplications.
epsilon  = 0.1;   %% The threshold for removing the poles in the epsilon-neighborhood of gamma, as gamma is in the first poles.
num_points = 80;  %% The number of ratios that you want to consider in the interval (0,1].
dominant_zp = ones(1,num_points); %% initialization

i = 0;
condition_counter = 0;
max_consecutive_iterations = 1;
previous_condition = false;

for ratio = linspace(0.01, 1, num_points)
i = i+1;
pwrq = pwrw;  
[Ws, Wd, Qs, Qd, one_vec, size_W] = create_W_Q(pwrw, pwrq, ratio);
Q = Qs; W = Ws; I = eye(size_W);

G_zinv        = (rho / (1/z * (1/z - 1))) * ((1/z - 1) * I + eta * Q) * inv((1/z - 1 + alpha) * I + eta * (1 + (alpha / (1/z - gamma))) * Q) *  ((1/z - 1) * I + eta * W);
G_z           = (rho / (z * (z - 1))) * ((z - 1) * I + eta * Q) * inv((z - 1 + alpha) * I + eta * (1 + (alpha / (z - gamma))) * Q) * ((z - 1) * I + eta * W);
G__t_z        = G_z.';
S_beta_z      = ((1-L*mu)*(2-z-1/z)-zeta+beta)*I + z*W + (1/z)*W.' +zeta*(one_vec*one_vec.');
G_zinv_simp   = simplify(G_zinv);
G__t_z_simp   = simplify(G__t_z);
S_beta_z_simp = simplify(S_beta_z);
Omega         = S_beta_z_simp+G_zinv_simp+G__t_z_simp;
E_z           = (z-1)/(eta*(gamma-1))/(z-gamma) * ((z-1+alpha)*(z-gamma)*I + (eta*(z-gamma+alpha))*Q);
E_zinv        = (1/z-1)/(eta*(gamma-1))/(1/z-gamma) * ((1/z-1+alpha)*(1/z-gamma)*I + (eta*(1/z-gamma+alpha))*Q);
E__t_z        = E_z.';

[rootsn_omega, rootsd_omega] = find_roots_of_det_of(Omega, z);
[rootsn_Ez, rootsd_Ez] = find_roots_of_det_of(E__t_z, z);
[rootsn_Ezinv, rootsd_Ezinv] = find_roots_of_det_of(E_zinv, z);

all_roots = [rootsn_Ez.', rootsn_Ezinv.', rootsn_omega.'];
all_roots = remove_root_gamma(all_roots, gamma, epsilon); 

condition1 = any(imag(rootsn_omega) > thr_imag   & (abs(rootsn_omega) == 1));
condition2 = any(imag(rootsn_Ez)> thr_imag       & (abs(rootsn_Ez) == 1));
condition3 = any(imag(rootsn_Ezinv)> thr_imag    & (abs(rootsn_Ezinv) == 1));

if condition1 || condition2 || condition3
    disp (['For ratio=', num2str(ratio), ': complex pole on a unit circle exists that make the determinant zero.'])
    dominant_zp(i) = 1.;
else
    dom_zp = find_dominant(all_roots);    
    disp (['For ratio=', num2str(ratio), ': the dominant pole is:', num2str(double(dom_zp))])
    dominant_zp(i) = double(abs(dom_zp));
end
end


%% Create the plot

figure;
plot(linspace(0.01, 1, num_points), dominant_zp, '-o', 'LineWidth', 5, 'MarkerSize', 0.1); 

ax = gca;
ax.FontSize = 28;  
ax.FontName = 'Times New Roman';  
ax.LineWidth = 3; 

xlabel('\bf{Ratio:} $n/N$', 'Interpreter', 'latex', 'FontSize', 28);
ylabel('\bf{Dominant Pole Magnitude}', 'Interpreter', 'latex', 'FontSize', 24);

ylim_current = ylim;
ylim([ylim_current(1), 1.03]);

grid on;
legend_text = sprintf('Power of weights = %.2f', pwrw);
legend(legend_text, 'Interpreter', 'latex', 'FontSize', 26, 'Location', 'best');
set(gcf, 'Color', 'w');

filename = sprintf('dom_zp_pwr_%.2f_alpha_%.2f_rho_%.2f_gamma_%.2f.png', pwrw, alpha, rho, gamma);
saveas(gcf, filename);
print(gcf, 'plot', '-dpng', '-r300');  

hold off;


%% Utility Functions

function all_roots_filtered = remove_root_gamma(all_roots, gamma, epsilon) 
    index_to_keep = abs(all_roots - gamma) >= epsilon;
    all_roots_filtered = all_roots(index_to_keep);
end


function dom_zp = find_dominant(roots)
magnitudes = abs(roots);
indices = find(magnitudes < 0.98);
elements_less_than_one = roots(indices);
[~, max_index] = max(abs(elements_less_than_one));
dom_zp = elements_less_than_one(max_index);
end


function [rootsn, rootsd] = find_roots_of_det_of(term, z)
    det_term = simplify(det(simplify(term)));
    [N, D]=numden(simplify(det_term));
    R1 = solve(N,z);
    R2 = solve(D,z);
    rootsn = vpa(R1);
    rootsd = vpa(R2);
end

function [Ws, Wd, Qs, Qd, one_vec, size_W] = create_W_Q(pwrw, pwrq, ratio)
    IB1=-(1-ratio);
    IB2=-sqrt(ratio*(1-ratio));
    IB3=-ratio;
    IZ=0;
    Ws=pwrw*[IB1+1 IB2 IZ IZ;
    IB2 IB3+2 IB3 IB2;
    IB2 IB3 IB3+2 IB2;
    IZ IZ IB2 IB1+1];
    Wd=pwrw*[1 0 0 0;
    0 2 0 0;
    0 0 2 0;
    0 0 0 1];
    Qs= pwrq*([IB1 IB2 IZ IZ;
    IB2 IB3 IB3 IB2;
    IB2 IB3 IB3 IB2;
    IZ IZ IB2 IB1]+(1+ratio)*eye(4));
    Qd=pwrq*(1+ratio)*eye(4);
    one_vec= [sqrt(1-ratio);
    sqrt(ratio);
    sqrt(ratio);
    sqrt(1-ratio)];
    ss=size(Ws);
    size_W=ss(1);  
   
% % I added this to have 11^T*11^T = 11^T
    one_vec = one_vec/sqrt(2);
end