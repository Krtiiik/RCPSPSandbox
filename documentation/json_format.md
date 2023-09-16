# JSON instance serialization format

## Instance

```
{
  "Name"      : string,
  "Horizon"   : int,
  "Resources" : array[Resource],
  "Projects"  : array[Project],
  "Jobs"      : array[Job]
}
```

## Resource

```
{
    "Type": ResourceType,
    "Id": int,
    "Capacity": int
}
```

### ResourceType

Resource type is a string with one of the following values representing the corresponding resource type:

- `"R"` -> Renewable
- `"N"` -> Non-Renewable
- `"D"` -> Doubly Constrained

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
    "Id": int,
    "Resource consumption": ResourceConsumption,
    "Successors": array[int]
}
```

### ResourceConsumption

```
{
    "Duration": int,
    "Consumptions": dict{string: int}
}
```

The `Consumptions` entry describes individual consumptions of specified resources. The dictionary is indexed by
keys of resources and the values are the corresponding consumptions of the resources.
