-- High risk claims
SELECT COUNT(*) 
FROM insurance_claims
WHERE claim_amount > 1500;

-- Spatial aggregation
SELECT claim_type, COUNT(*)
FROM insurance_claims
GROUP BY claim_type;

