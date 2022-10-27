# EmissionsForecastBatchParametersDTO


## Properties
Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**requested_at** | **datetime, none_type** | For historical forecast requests, this value is the timestamp used to access the most  recently generated forecast as of that time. | [optional] 
**location** | **str, none_type** | The location of the forecast | [optional] 
**data_start_at** | **datetime, none_type** | Start time boundary of forecasted data points.Ignores current forecast data points before this time.  Defaults to the earliest time in the forecast data. | [optional] 
**data_end_at** | **datetime, none_type** | End time boundary of forecasted data points. Ignores current forecast data points after this time.  Defaults to the latest time in the forecast data. | [optional] 
**window_size** | **int, none_type** | The estimated duration (in minutes) of the workload.  Defaults to the duration of a single forecast data point. | [optional] 

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


