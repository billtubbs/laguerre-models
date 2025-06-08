model Laguerre3 "Third-order dynamic model based on Laguerre polynomials"
  // Parameters
  parameter Real p = 0.5 "Laguerre pole parameter (0 < p < 1)";
  parameter Real K = 1.0 "System gain";
  parameter Real[3] c = {0.8, 0.5, 0.2} "Output coefficient vector C^T";
  parameter Real[3] l0 = {0.0, 0.0, 0.0} "Initial Laguerre state vector";
  
  // Variables
  Real[3] l(start=l0) "Laguerre state vector";
  Real u "Input signal";
  Real y "Output signal";
  
  // Laguerre basis functions for verification (optional)
  Real L0, L1, L2 "Laguerre basis functions";
  
  // Internal variables for matrix operations
  Real sqrt_2p "Square root of 2*p for normalization";
  
equation
  // Calculate normalization factor
  sqrt_2p = sqrt(2*p);
  
  // Laguerre network state-space representation
  // Based on Zervos & Dumont (1988) formulation
  // dx/dt = A*x + b*u where A is lower triangular Laguerre matrix
  
  // State equations for third-order Laguerre network
  der(l[1]) = -p * l[1] + sqrt_2p * K * u;
  der(l[2]) = sqrt_2p * p * l[1] - p * l[2];
  der(l[3]) = sqrt_2p * p * l[2] - p * l[3];
  
  // Output equation: y = C^T * l
  y = c[1] * l[1] + c[2] * l[2] + c[3] * l[3];
  
  // Laguerre basis functions (for reference and verification)
  L0 = sqrt_2p * exp(-p * time);
  L1 = sqrt_2p * (1 - 2*p*time) * exp(-p * time);
  L2 = sqrt_2p * (1 - 4*p*time + 2*p^2*time^2) * exp(-p * time);
  
  // Example input signal (step input at t=1)
  u = if time >= 1.0 then 1.0 else 0.0;
  
  annotation(
    experiment(StartTime=0, StopTime=20, Tolerance=1e-6),
    Documentation(info="<html>
      <p>This model implements a third-order dynamic system based on Laguerre series representation 
      following the approach of Zervos & Dumont (1988):</p>
      <ul>
        <li>Laguerre pole parameter p = 0.5 (controls decay rate)</li>
        <li>System gain K = 1.0</li>
        <li>Output coefficients C^T = [0.8, 0.5, 0.2]</li>
        <li>Step input applied at t = 1.0 s</li>
      </ul>
      
      <p><b>Theory:</b></p>
      <p>The Laguerre-based model represents the system impulse response as:</p>
      <p>g(t) = Σ c_i * L_i(t)</p>
      
      <p>Where L_i(t) are orthonormal Laguerre functions:</p>
      <p>L_i(t) = √(2p) * L_i^(0)(2pt) * e^(-pt)</p>
      
      <p>The state-space representation is:</p>
      <p>dl/dt = A*l + b*u</p>
      <p>y = C^T * l</p>
      
      <p>Where A is the lower triangular Laguerre matrix and b is the input vector.</p>
      
      <p><b>Advantages:</b></p>
      <ul>
        <li>Reduced model order compared to conventional approaches</li>
        <li>Good approximation with fewer parameters</li>
        <li>Suitable for systems with dominant time constants</li>
        <li>Robust to parameter variations</li>
      </ul>
      
      <p><b>Applications:</b> Process control, adaptive control, system identification</p>
    </html>")
  );
end Laguerre3;
