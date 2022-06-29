function y=check_ascending(mu,risk)
    
    R = corrcoef(mu,risk);
    R = R(1,2);
    y = true;
    if R >= 0.8
        y= true;
    else
        y= false;

end