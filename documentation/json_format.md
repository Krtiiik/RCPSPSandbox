# JSON instance serialization format

This document describes the JSON format of serializes instances. Fields marked with a prepended `+` are present only
in the extended instance format - they are not part of the original PSPLib instance specification.

## Instance

```
{
  "Name"        : string,
  "Horizon"     : int,
  "Resources"   : array[Resource],
  "Projects"    : array[Project],
  "Jobs"        : array[Job],
  +"Components" : array[Component]
}
```

## Resource

```
{
    "Type"          : ResourceType,
    "Id"            : int,
    "Capacity"      : int,
    +"Availability" : array[AvailabilityInterval]
}
```

### ResourceType

Resource type is a string with one of the following values representing the corresponding resource type:

- `"R"` -> Renewable
- `"N"` -> Non-Renewable
- `"D"` -> Doubly Constrained

### +AvailabilityInterval

Availability interval defines an interval on which a related resource is available. The values represent day-hours
within a standard day. Capacity can be specified for the interval which, on that interval, overrides the default
capacity of the resource.

```
{
    "Start"      : int
    "End"        : int
    ("Capacity") : int
}
```

## Project

```
{
    "Id"             : int,
    "Due date"       : int,
    "Tardiness cost" : int
}
```

## Job

```
{
    "Id"                   : int,
    "Duration"             : int,
    "Resource consumption" : ResourceConsumption,
    "Successors"           : array[int],
    +"Due date"            : int,
    +"Completed            : bool
}
```

### ResourceConsumption

```
{
    "Consumptions": dict{string: int}
}
```

The `Consumptions` entry describes individual consumptions of specified resources. The dictionary is indexed by
keys of resources and the values are the corresponding consumptions of the resources.
