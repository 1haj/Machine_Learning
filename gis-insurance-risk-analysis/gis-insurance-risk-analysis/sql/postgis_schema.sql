CREATE TABLE insurance_claims (
    claim_id INT PRIMARY KEY,
    claim_amount FLOAT,
    claim_type TEXT,
    policy_type TEXT,
    incident_date DATE,
    geom GEOGRAPHY(Point, 4326)
);
