-- Auto-generated MGB extraction for NCT-MGB-SEPSIS-STEROID (tteEngine #103 codegen).
-- DO NOT EDIT BY HAND — regenerate from the ExtractionPlan.
-- GATED: review + run on MGB only with data-access authorization. Output is
-- the canonical 5-col stream; feed it to tteEngine.adapters.mgb.extract or
-- use directly (already canonical).
-- Extraction window: [-48.0h, 24.0h] relative to each trajectory's first event.

WITH cohort AS (
    SELECT DISTINCT TRAJECTORY_ID
    FROM MGB.CANONICAL_EVENTS
    WHERE EVENT_TYPE = 'diagn'
      AND EVENT_NAME IN ('0380', '03810', '03811', '03812', '03819', '0382', '0383', '03840', '03841', '03842', '03843', '03844', '03849', '0388', '0389', '78552', '99591', '99592', 'A40', 'A400', 'A401', 'A403', 'A408', 'A409', 'A41', 'A410', 'A411', 'A412', 'A413', 'A414', 'A415', 'A4150', 'A4151', 'A4152', 'A4153', 'A4159', 'A418', 'A4181', 'A4189', 'A419', 'A499', 'R6520', 'R6521', 'R7881')
),
windowed AS (
    SELECT e.TRAJECTORY_ID, e.TIMESTAMP, e.EVENT_TYPE, e.EVENT_NAME, e.EVENT_VALUE,
           MIN(e.TIMESTAMP) OVER (PARTITION BY e.TRAJECTORY_ID) AS T0
    FROM MGB.CANONICAL_EVENTS e
    JOIN cohort USING (TRAJECTORY_ID)
)
SELECT TRAJECTORY_ID, TIMESTAMP, EVENT_TYPE, EVENT_NAME, EVENT_VALUE
FROM windowed
WHERE TIMESTAMP BETWEEN DATEADD('hour', -48.0, T0) AND DATEADD('hour', 24.0, T0)
  AND (
        (EVENT_TYPE = 'diagn' AND EVENT_NAME IN ('0380', '03810', '03811', '03812', '03819', '0382', '0383', '03840', '03841', '03842', '03843', '03844', '03849', '0388', '0389', '78552', '99591', '99592', 'A40', 'A400', 'A401', 'A403', 'A408', 'A409', 'A41', 'A410', 'A411', 'A412', 'A413', 'A414', 'A415', 'A4150', 'A4151', 'A4152', 'A4153', 'A4159', 'A418', 'A4181', 'A4189', 'A419', 'A499', 'R6520', 'R6521', 'R7881'))
     OR (EVENT_TYPE = 'medic' AND EVENT_NAME IN ('dexamethasone', 'fludrocortisone', 'hydrocortisone', 'methylprednisolone', 'prednisolone', 'prednisone'))
     OR (EVENT_TYPE = 'outco' AND EVENT_NAME IN ('death'))
  )
ORDER BY TRAJECTORY_ID, TIMESTAMP;
